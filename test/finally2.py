import json
import os
import pickle
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, CLIPModel, CLIPProcessor

# ----------- MUST HAVE (GT mask) -----------
try:
    from pycocotools.coco import COCO
except Exception as e:
    raise RuntimeError(
        "Need pycocotools for GT masks. In PyCharm Interpreter install: pycocotools (or pycocotools-windows)."
    ) from e


# =========================
# 0) PATHS (edit here)
# =========================
REFCOCO_ROOT = r"E:\下载\fdb5e-main\fdb5e-main\refcoco\refcoco"
INSTANCES_JSON = os.path.join(REFCOCO_ROOT, "instances.json")
REFS_PKL = os.path.join(REFCOCO_ROOT, "refs(unc).p")  # you can switch to refs(google).p

# COCO images: choose ONE or keep both (it will search in order)
COCO_IMAGE_ROOTS = [
    r"E:\Download 百度网盘\COCO 2014\train2014\train2014",
    # r"E:\Download 百度网盘\COCO 2014\val2014\val2014",
]

# Local models (from your screenshots)
VLM_DIR = r"E:\Vision1\vision"          # 图5
CLIP_DIR = r"E:\2026\CLIP_patch32"      # 图3
SAM_CKPT = r"E:\2026\SAM1\sam_vit_b_01ec64.pth"  # 图4

# Output
OUT_DIR = r"E:\2026\refcoco_bench_out"


# =========================
# 1) CONFIG
# =========================
@dataclass
class Cfg:
    name: str = "BASE"

    # how many questions to test (keep small to avoid black screen)
    num_samples: int = 10
    seed: int = 0

    # --- SAM candidates ---
    points_per_side: int = 24
    pred_iou_thresh: float = 0.88
    stability_thresh: float = 0.92
    min_mask_area: int = 250
    min_area_frac: float = 0.0015
    max_area_frac: float = 0.85
    max_cands: int = 240  # prefilter top-N by (pred_iou + stability)

    # --- CLIP ranking ---
    bbox_pad: int = 16
    clip_batch: int = 64
    topk_for_vlm: int = 12  # show topk to VLM

    # --- VLM critic ---
    use_vlm: bool = True  # outer loop: 0/1 iteration (keep minimal)
    save_debug_images: bool = False

    # --- refine (SAM predictor) ---
    refine_max_iters: int = 4
    refine_stop_iou: float = 0.985  # between consecutive masks

    # speed knobs
    amp: bool = True  # autocast
    tf32: bool = True


BASE = Cfg(
    name="BASE",
    num_samples=10,
    points_per_side=24,
    max_cands=240,
    topk_for_vlm=12,
    refine_max_iters=4,
    use_vlm=True,
)

# Acceleration strategy (at least 1):
# 1) Reduce SAM candidates: points_per_side 24->16, max_cands 240->120
# 2) Reduce TopK to VLM: 12->8
# 3) Reduce refine iters: 4->2
FAST = Cfg(
    name="FAST",
    num_samples=10,
    points_per_side=16,
    max_cands=120,
    topk_for_vlm=8,
    refine_max_iters=2,
    use_vlm=True,
)


# =========================
# 2) UTIL: metrics & counters
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter) / float(union + 1e-8)


@dataclass
class Counters:
    # tool calls
    sam_amg_calls: int = 0
    sam_predict_calls: int = 0
    vlm_calls: int = 0
    clip_text_calls: int = 0
    clip_image_batches: int = 0


@dataclass
class OneResult:
    cfg: str
    ref_id: Any
    image_id: int
    ann_id: int
    expr: str
    image_path: str

    iou: float
    success_05: bool
    outer_iters: int
    refine_iters: int

    time_total: float
    t_sam: float
    t_clip: float
    t_vlm: float
    t_refine: float

    peak_alloc_mb: float
    peak_reserved_mb: float

    counters: Dict[str, int]


def summarize(rows: List[OneResult]) -> Dict[str, Any]:
    ious = [r.iou for r in rows]
    succ = [1.0 if r.success_05 else 0.0 for r in rows]
    return {
        "num": len(rows),
        "mIoU": float(np.mean(ious)) if ious else None,
        "success@0.5": float(np.mean(succ)) if succ else None,
        "avg_outer_iters": float(np.mean([r.outer_iters for r in rows])) if rows else None,
        "avg_refine_iters": float(np.mean([r.refine_iters for r in rows])) if rows else None,
        "mean_time_s": float(np.mean([r.time_total for r in rows])) if rows else None,
        "mean_peak_alloc_mb": float(np.mean([r.peak_alloc_mb for r in rows])) if rows else None,
        "mean_peak_reserved_mb": float(np.mean([r.peak_reserved_mb for r in rows])) if rows else None,
        "sum_tool_calls": {
            "sam_amg_calls": int(sum(r.counters["sam_amg_calls"] for r in rows)),
            "sam_predict_calls": int(sum(r.counters["sam_predict_calls"] for r in rows)),
            "vlm_calls": int(sum(r.counters["vlm_calls"] for r in rows)),
            "clip_text_calls": int(sum(r.counters["clip_text_calls"] for r in rows)),
            "clip_image_batches": int(sum(r.counters["clip_image_batches"] for r in rows)),
        },
    }


# =========================
# 3) LOAD DATA: RefCOCO refs + COCO instances
# =========================
def load_refcoco_samples(refs_pkl: str, coco: COCO, roots: List[str],
                         num_samples: int, seed: int) -> List[Dict[str, Any]]:
    obj = pickle.load(open(refs_pkl, "rb"))
    refs = obj["refs"] if isinstance(obj, dict) and "refs" in obj else obj

    rng = random.Random(seed)
    rng.shuffle(refs)

    out = []
    for r in refs:
        image_id = int(r["image_id"])
        ann_id = int(r["ann_id"])
        ref_id = r.get("ref_id", None)

        sents = r.get("sentences", [])
        if not sents:
            continue
        sent0 = sents[0]
        expr = sent0.get("sent", None) if isinstance(sent0, dict) else str(sent0)
        if not expr:
            continue

        coco_img = coco.loadImgs([image_id])[0]
        try:
            img_path = resolve_image_path(coco_img, roots)
        except FileNotFoundError:
            continue  # 跳过你本地没有图的样本

        out.append({
            "ref_id": ref_id,
            "image_id": image_id,
            "ann_id": ann_id,
            "expr": str(expr),
            "image_path": img_path,
        })

        if len(out) >= num_samples:
            break

    if not out:
        raise RuntimeError("No samples found that exist in your COCO_IMAGE_ROOTS. Check your train2014 path.")
    return out



def resolve_image_path(coco_img: Dict[str, Any], roots: List[str]) -> str:
    """
    SAFETY: only use COCO official file_name, e.g. COCO_train2014_000000xxxxxx.jpg
    Do NOT fallback to 000000xxxxxx.jpg (that causes wrong-image mismatch).
    """
    file_name = coco_img.get("file_name", "")
    if not file_name:
        raise FileNotFoundError(f"COCO image has no file_name. img_id={coco_img.get('id')}")

    for root in roots:
        p1 = os.path.join(root, file_name)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(root, os.path.basename(file_name))
        if os.path.exists(p2):
            return p2

    raise FileNotFoundError(f"Cannot find image file_name={file_name} under roots={roots}")


# =========================
# 4) MODELS: VLM / CLIP / SAM
# =========================
class VLM:
    def __init__(self, model_dir: str, counters: Counters):
        self.counters = counters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def pick_idx(self, review_img: Image.Image, expr: str, k: int) -> int:
        # Ask VLM to output {"pick_idx": n} only
        prompt = (
            "You are selecting the best segmentation candidate.\n"
            "In the image, the bottom area shows K candidate tiles labeled idx=0..K-1 with red mask overlays.\n"
            f"Expression: {expr}\n"
            f"Return ONLY JSON like {{\"pick_idx\": 0}} where pick_idx is in [0, {k-1}]."
        )

        self.counters.vlm_calls += 1

        if hasattr(self.proc, "apply_chat_template"):
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.proc(text=[text], images=[review_img], return_tensors="pt", padding=True)
        else:
            inputs = self.proc(text=[prompt], images=[review_img], return_tensors="pt", padding=True)

        if self.device == "cuda":
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        out = self.model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1)
        gen_ids = out[:, inputs["input_ids"].shape[1]:] if "input_ids" in inputs else out
        txt = self.proc.batch_decode(gen_ids, skip_special_tokens=True)[0]

        m = re.search(r"pick_idx\"?\s*:\s*(\d+)", txt)
        if m:
            return int(m.group(1))
        m2 = re.search(r"(\d+)", txt)
        return int(m2.group(1)) if m2 else 0


class ClipRanker:
    def __init__(self, clip_dir: str, counters: Counters):
        self.counters = counters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        p = Path(clip_dir)
        if not p.exists():
            raise FileNotFoundError(f"CLIP_DIR not found: {clip_dir}")

        # 自动判断权重格式：safetensors 或 pytorch bin
        has_st = any(p.glob("*.safetensors")) or (p / "model.safetensors").exists()
        has_bin = (p / "pytorch_model.bin").exists() or any(p.glob("pytorch_model*.bin"))

        if not has_st and not has_bin:
            raise OSError(f"No CLIP weights found in {clip_dir}. Files={ [x.name for x in p.glob('*')] }")

        # 关键：明确告诉 transformers 用哪种格式
        self.model = CLIPModel.from_pretrained(
            clip_dir,
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=bool(has_st),  # 目录没有 safetensors 就会自动用 bin
        ).to(self.device)

        self.proc = CLIPProcessor.from_pretrained(clip_dir, local_files_only=True)
        self.model.eval()

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        self.counters.clip_text_calls += 1
        inp = self.proc(text=[text], images=None, return_tensors="pt", padding=True)
        inp = {k: v.to(self.device) for k, v in inp.items() if hasattr(v, "to")}
        feat = self.model.get_text_features(**inp)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        return feat  # [1, d]

    @torch.inference_mode()
    def encode_images(self, images: List[Image.Image], batch: int, use_amp: bool) -> torch.Tensor:
        feats = []
        for i in range(0, len(images), batch):
            self.counters.clip_image_batches += 1
            ims = images[i:i + batch]
            inp = self.proc(text=None, images=ims, return_tensors="pt", padding=True)
            inp = {k: v.to(self.device) for k, v in inp.items() if hasattr(v, "to")}
            if use_amp and self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    f = self.model.get_image_features(**inp)
            else:
                f = self.model.get_image_features(**inp)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            feats.append(f)
        return torch.cat(feats, dim=0)  # [N, d]

    @staticmethod
    def sims(img_feats: torch.Tensor, text_feat: torch.Tensor) -> np.ndarray:
        s = (img_feats @ text_feat.T).detach().float().cpu().numpy().reshape(-1)
        return s


# =========================
# 5) SAM candidates + refine
# =========================
@dataclass
class Cand:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # xyxy
    score: float


def bbox_xywh_to_xyxy(b: List[float]) -> Tuple[int, int, int, int]:
    x, y, w, h = b
    return int(x), int(y), int(x + w), int(y + h)


def crop_bbox(img_rgb: np.ndarray, bbox: Tuple[int, int, int, int], pad: int) -> Image.Image:
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)
    return Image.fromarray(img_rgb[y1:y2 + 1, x1:x2 + 1])


def build_panel(img_rgb: np.ndarray, cands: List[Cand], pad: int, tile: int = 256) -> Image.Image:
    # show K tiles with red mask overlay + idx label
    K = len(cands)
    cols = 4 if K >= 8 else 2
    rows = int(np.ceil(K / cols))
    panel = Image.new("RGB", (cols * tile, rows * tile), (30, 30, 30))

    for i, c in enumerate(cands):
        x1, y1, x2, y2 = c.bbox
        H, W = img_rgb.shape[:2]
        x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
        x2p = min(W - 1, x2 + pad); y2p = min(H - 1, y2 + pad)

        crop = img_rgb[y1p:y2p + 1, x1p:x2p + 1].copy()
        m = c.mask[y1p:y2p + 1, x1p:x2p + 1].astype(np.uint8) * 255

        base = Image.fromarray(crop).convert("RGBA")
        red = Image.new("RGBA", base.size, (255, 0, 0, 120))
        alpha = Image.fromarray(m).convert("L")
        over = Image.composite(red, base, alpha).convert("RGB")

        d = ImageDraw.Draw(over)
        d.rectangle([0, 0, 220, 28], fill=(0, 0, 0))
        d.text((8, 6), f"idx={i}  s={c.score:.3f}", fill=(255, 255, 255))

        over = over.resize((tile, tile), resample=Image.BILINEAR)
        r = i // cols
        cc = i % cols
        panel.paste(over, (cc * tile, r * tile))

    return panel


def stack_review(original: Image.Image, panel: Image.Image, max_w: int = 1024) -> Image.Image:
    o = original.convert("RGB")
    p = panel.convert("RGB")
    if o.width > max_w:
        nh = int(o.height * (max_w / o.width))
        o = o.resize((max_w, nh), resample=Image.BILINEAR)
    if p.width != o.width:
        nh = int(p.height * (o.width / p.width))
        p = p.resize((o.width, nh), resample=Image.BILINEAR)
    out = Image.new("RGB", (o.width, o.height + p.height), (0, 0, 0))
    out.paste(o, (0, 0))
    out.paste(p, (0, o.height))
    return out


def load_sam_and_build_amg(ckpt: str, cfg: Cfg):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    def pick_type(p: str) -> str:
        n = Path(p).name.lower()
        if "vit_h" in n: return "vit_h"
        if "vit_l" in n: return "vit_l"
        return "vit_b"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_type = pick_type(ckpt)
    sam = sam_model_registry[sam_type](checkpoint=ckpt).to(device=device)

    amg = SamAutomaticMaskGenerator(
        sam,
        points_per_side=cfg.points_per_side,
        pred_iou_thresh=cfg.pred_iou_thresh,
        stability_score_thresh=cfg.stability_thresh,
        min_mask_region_area=cfg.min_mask_area,
    )
    return sam, amg, SamPredictor


def refine_iterative(predictor, init_mask: np.ndarray, init_bbox: Tuple[int, int, int, int],
                     counters: Counters, max_iters: int, stop_iou: float) -> Tuple[np.ndarray, int]:
    # Very small iterative refine: use box + sampled points from previous mask
    H, W = init_mask.shape
    x1, y1, x2, y2 = init_bbox
    box = np.array([x1, y1, x2, y2], dtype=np.float32)

    prev = init_mask.astype(np.uint8)
    used = 0

    for t in range(1, max_iters + 1):
        used = t
        ys, xs = np.where(prev > 0)
        if len(xs) == 0:
            pts = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float32)
            labs = np.array([1], dtype=np.int32)
        else:
            # sample up to 8 positive points
            k = min(8, len(xs))
            idx = np.random.choice(len(xs), size=k, replace=False)
            pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
            labs = np.ones((k,), dtype=np.int32)

        counters.sam_predict_calls += 1
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=labs,
            box=box,
            multimask_output=True,
        )
        cur = masks[int(np.argmax(scores))].astype(np.uint8)
        if mask_iou(prev, cur) >= stop_iou:
            return cur, used
        prev = cur

    return prev, used


# =========================
# 6) RUN ONE SAMPLE
# =========================
def run_one(expr: str, image_path: str, gt_mask: np.ndarray,
            clip: ClipRanker, vlm: Optional[VLM],
            amg, sam, SamPredictor,
            cfg: Cfg, counters: Counters) -> Tuple[float, bool, int, int, Dict[str, float]]:
    t0 = time.perf_counter()

    img = Image.open(image_path).convert("RGB")
    img_rgb = np.array(img)
    H, W = img_rgb.shape[:2]

    # --- SAM candidates ---
    t_sam0 = time.perf_counter()
    counters.sam_amg_calls += 1
    amg_out = amg.generate(img_rgb)
    t_sam = time.perf_counter() - t_sam0

    cands0 = []
    for d in amg_out:
        m = d["segmentation"].astype(np.uint8)
        af = float(m.sum()) / float(H * W)
        if af < cfg.min_area_frac or af > cfg.max_area_frac:
            continue
        bbox = bbox_xywh_to_xyxy(d.get("bbox", [0, 0, 0, 0]))
        # pre-score for sorting (pred_iou + stability)
        pre = float(d.get("predicted_iou", 0.0) or 0.0) + float(d.get("stability_score", 0.0) or 0.0)
        cands0.append((pre, m, bbox))

    if not cands0:
        return 0.0, False, 0, 0, {"t_sam": t_sam, "t_clip": 0.0, "t_vlm": 0.0, "t_refine": 0.0, "t_total": time.perf_counter() - t0}

    cands0.sort(key=lambda x: x[0], reverse=True)
    cands0 = cands0[:cfg.max_cands]

    # --- CLIP rank ---
    t_clip0 = time.perf_counter()
    text_feat = clip.encode_text(expr)
    crops = [crop_bbox(img_rgb, bb, cfg.bbox_pad) for _, _, bb in cands0]
    img_feats = clip.encode_images(crops, batch=cfg.clip_batch, use_amp=cfg.amp)
    sims = clip.sims(img_feats, text_feat)

    # topK for VLM
    idx_sorted = np.argsort(-sims)
    K = min(cfg.topk_for_vlm, len(idx_sorted))
    top_idx = idx_sorted[:K].tolist()

    top_cands = [Cand(mask=cands0[i][1], bbox=cands0[i][2], score=float(sims[i])) for i in top_idx]
    t_clip = time.perf_counter() - t_clip0

    # --- (optional) VLM pick among topK ---
    t_vlm = 0.0
    outer_iters = 0
    pick = 0
    if cfg.use_vlm and vlm is not None and len(top_cands) > 1:
        outer_iters = 1
        t_v0 = time.perf_counter()
        panel = build_panel(img_rgb, top_cands, pad=cfg.bbox_pad, tile=256)
        review = stack_review(img, panel, max_w=1024)
        if cfg.save_debug_images:
            Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
            review.save(os.path.join(OUT_DIR, f"review_{Path(image_path).stem}.jpg"))
        pick = int(np.clip(vlm.pick_idx(review, expr, k=len(top_cands)), 0, len(top_cands) - 1))
        t_vlm = time.perf_counter() - t_v0

    chosen = top_cands[pick]

    # --- refine ---
    t_r0 = time.perf_counter()
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)

    refined, refine_iters = refine_iterative(
        predictor,
        init_mask=chosen.mask,
        init_bbox=chosen.bbox,
        counters=counters,
        max_iters=cfg.refine_max_iters,
        stop_iou=cfg.refine_stop_iou,
    )
    t_refine = time.perf_counter() - t_r0

    iou = mask_iou(refined, gt_mask)
    ok = (iou >= 0.5)

    t_total = time.perf_counter() - t0
    times = {"t_sam": t_sam, "t_clip": t_clip, "t_vlm": t_vlm, "t_refine": t_refine, "t_total": t_total}
    return float(iou), bool(ok), int(outer_iters), int(refine_iters), times


# =========================
# 7) BENCHMARK
# =========================
def run_benchmark(cfg: Cfg) -> Tuple[Dict[str, Any], List[OneResult]]:
    set_seed(cfg.seed)
    out_dir = Path(OUT_DIR) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # load data
    coco = COCO(INSTANCES_JSON)
    samples = load_refcoco_samples(REFS_PKL, coco, COCO_IMAGE_ROOTS, num_samples=cfg.num_samples, seed=cfg.seed)

    # counters + models
    counters = Counters()

    print(f"\n[{cfg.name}] Loading models ...")
    clip = ClipRanker(CLIP_DIR, counters)
    vlm = VLM(VLM_DIR, counters) if cfg.use_vlm else None
    sam, amg, SamPredictor = load_sam_and_build_amg(SAM_CKPT, cfg)
    print(f"[{cfg.name}] Models loaded.")

    rows: List[OneResult] = []

    for i, s in enumerate(samples, 1):
        ref_id, image_id, ann_id, expr = s["ref_id"], s["image_id"], s["ann_id"], s["expr"]

        # image path
        img_path = s["image_path"]
        # GT mask
        ann = coco.loadAnns([ann_id])[0]
        gt_mask = coco.annToMask(ann).astype(np.uint8)  # [H,W] 0/1

        # VRAM peak per sample
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()

        iou, ok, outer_iters, refine_iters, times = run_one(
            expr=expr,
            image_path=img_path,
            gt_mask=gt_mask,
            clip=clip,
            vlm=vlm,
            amg=amg,
            sam=sam,
            SamPredictor=SamPredictor,
            cfg=cfg,
            counters=counters,
        )

        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_resv = torch.cuda.max_memory_reserved() / (1024 ** 2)
        else:
            peak_alloc = 0.0
            peak_resv = 0.0

        row = OneResult(
            cfg=cfg.name,
            ref_id=ref_id,
            image_id=image_id,
            ann_id=ann_id,
            expr=expr,
            image_path=img_path,
            iou=float(iou),
            success_05=bool(ok),
            outer_iters=int(outer_iters),
            refine_iters=int(refine_iters),
            time_total=float(times["t_total"]),
            t_sam=float(times["t_sam"]),
            t_clip=float(times["t_clip"]),
            t_vlm=float(times["t_vlm"]),
            t_refine=float(times["t_refine"]),
            peak_alloc_mb=float(peak_alloc),
            peak_reserved_mb=float(peak_resv),
            counters=asdict(counters),
        )
        rows.append(row)

        print(f"[{cfg.name}] {i}/{len(samples)}  "
              f"IoU={iou:.3f} ok={ok}  "
              f"time={times['t_total']:.3f}s (sam={times['t_sam']:.3f}, clip={times['t_clip']:.3f}, vlm={times['t_vlm']:.3f}, ref={times['t_refine']:.3f})  "
              f"peakMB={peak_alloc:.1f}")

        # save incremental jsonl
        with open(out_dir / "rows.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    summ = summarize(rows)
    pack = {"cfg": asdict(cfg), "summary": summ}
    (out_dir / "summary.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{cfg.name}] SUMMARY:\n{json.dumps(summ, ensure_ascii=False, indent=2)}")
    return summ, rows


def compare(base_sum: Dict[str, Any], fast_sum: Dict[str, Any]) -> Dict[str, Any]:
    # show impact of acceleration strategy
    def safe_ratio(a, b):
        return None if (a is None or b is None or b == 0) else float(a) / float(b)

    out = {
        "speedup_x": safe_ratio(base_sum.get("mean_time_s"), fast_sum.get("mean_time_s")),
        "mIoU_drop": None if (base_sum.get("mIoU") is None or fast_sum.get("mIoU") is None) else float(fast_sum["mIoU"] - base_sum["mIoU"]),
        "success_drop": None if (base_sum.get("success@0.5") is None or fast_sum.get("success@0.5") is None) else float(fast_sum["success@0.5"] - base_sum["success@0.5"]),
        "peak_alloc_mb_change": None if (base_sum.get("mean_peak_alloc_mb") is None or fast_sum.get("mean_peak_alloc_mb") is None) else float(fast_sum["mean_peak_alloc_mb"] - base_sum["mean_peak_alloc_mb"]),
        "tool_calls_change": {
            k: int(fast_sum["sum_tool_calls"][k] - base_sum["sum_tool_calls"][k])
            for k in base_sum["sum_tool_calls"].keys()
        }
    }
    return out


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # run BASE then FAST on same sample count/seed
    base_sum, _ = run_benchmark(BASE)
    fast_sum, _ = run_benchmark(FAST)

    impact = compare(base_sum, fast_sum)
    (Path(OUT_DIR) / "compare_base_vs_fast.json").write_text(
        json.dumps(impact, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== ACCELERATION IMPACT (BASE -> FAST) ===")
    print(json.dumps(impact, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
