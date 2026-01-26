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
REFS_PKL = os.path.join(REFCOCO_ROOT, "refs(unc).p")  # or refs(google).p

COCO_IMAGE_ROOTS = [
    r"E:\Download 百度网盘\COCO 2014\train2014\train2014",
    # r"E:\Download 百度网盘\COCO 2014\val2014\val2014",
]

VLM_DIR = r"E:\Vision1\vision"
CLIP_DIR = r"E:\2026\CLIP_patch32"
SAM_CKPT = r"E:\2026\SAM1\sam_vit_b_01ec64.pth"

OUT_DIR = r"E:\2026\5out"


# =========================
# 1) CONFIG
# =========================
@dataclass
class Cfg:
    name: str = "BASE"

    # how many questions to test (keep small to avoid black screen)
    num_samples: int = 3
    seed: int = 0

    # --- SAM candidates (AMG) ---
    points_per_side: int = 24
    pred_iou_thresh: float = 0.88
    stability_thresh: float = 0.92
    min_mask_area: int = 250
    min_area_frac: float = 0.0015
    max_area_frac: float = 0.85
    max_cands: int = 240  # keep top-N by (pred_iou + stability)

    # --- CLIP ranking ---
    bbox_pad: int = 16
    clip_batch: int = 64
    topk_for_vlm: int = 12  # show topk to VLM
    use_margin_posneg: bool = True  # score = max(pos) - max(neg)

    # --- TRUE closed-loop (2~3 iterations) ---
    outer_max_iters: int = 3
    force_vlm_each_iter: bool = True
    vlm_gap_trigger: float = 0.02
    init_pos_is_expr: bool = True
    init_neg_bank: bool = True

    # --- VLM reviewer ---
    use_vlm: bool = True
    save_debug_images: bool = True
    vlm_tile: int = 224
    vlm_max_width: int = 896
    vlm_max_new_tokens: int = 128  # longer = slower; 64/96 can be faster

    # --- refine (SAM predictor) ---
    refine_max_iters: int = 4
    refine_stop_iou: float = 0.985

    # speed knobs
    amp: bool = True
    tf32: bool = True


BASE = Cfg(
    name="BASE",
    num_samples=3,
    points_per_side=24,
    max_cands=240,
    topk_for_vlm=12,
    refine_max_iters=4,
    use_vlm=True,
    outer_max_iters=3,
    force_vlm_each_iter=True,
    save_debug_images=True,
)

FAST = Cfg(
    name="FAST",
    num_samples=3,
    points_per_side=16,
    max_cands=120,
    topk_for_vlm=8,
    refine_max_iters=2,
    use_vlm=True,
    outer_max_iters=2,
    force_vlm_each_iter=False,
    vlm_gap_trigger=0.03,
    save_debug_images=True,
)


# =========================
# 2) UTIL: seed / metrics / counters
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
    sam_amg_calls: int = 0
    sam_predict_calls: int = 0
    vlm_calls: int = 0
    clip_text_calls: int = 0          # encode_texts calls
    clip_image_batches: int = 0       # encode_images batch forward count
    clip_score_calls: int = 0         # each scoring round counts 1

    def snapshot(self) -> Dict[str, int]:
        return asdict(self)

    @staticmethod
    def delta(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
        return {k: int(after.get(k, 0) - before.get(k, 0)) for k in after.keys()}


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

    tool_calls: Dict[str, int]  # per-sample delta


def summarize(rows: List[OneResult]) -> Dict[str, Any]:
    ious = [r.iou for r in rows]
    succ = [1.0 if r.success_05 else 0.0 for r in rows]
    tool_sum = {}
    if rows:
        keys = rows[0].tool_calls.keys()
        tool_sum = {k: int(sum(r.tool_calls.get(k, 0) for r in rows)) for k in keys}

    return {
        "num": len(rows),
        "mIoU": float(np.mean(ious)) if ious else None,
        "success@0.5": float(np.mean(succ)) if succ else None,
        "avg_outer_iters": float(np.mean([r.outer_iters for r in rows])) if rows else None,
        "avg_refine_iters": float(np.mean([r.refine_iters for r in rows])) if rows else None,
        "mean_time_s": float(np.mean([r.time_total for r in rows])) if rows else None,
        "mean_peak_alloc_mb": float(np.mean([r.peak_alloc_mb for r in rows])) if rows else None,
        "mean_peak_reserved_mb": float(np.mean([r.peak_reserved_mb for r in rows])) if rows else None,
        "sum_tool_calls": tool_sum,
    }


# =========================
# 3) DATA: RefCOCO refs + COCO instances
# =========================
def resolve_image_path(coco_img: Dict[str, Any], roots: List[str]) -> str:
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
            continue

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
        raise RuntimeError("No samples found that exist in your COCO_IMAGE_ROOTS. Check your COCO2014 path.")
    return out


# =========================
# 4) VISUAL HELPERS
# =========================
def _mask_to_pil(mask01: np.ndarray) -> Image.Image:
    m = (mask01.astype(np.uint8) * 255)
    return Image.fromarray(m, mode="L")


def overlay_mask(img: Image.Image, mask01: np.ndarray, color=(255, 0, 0), alpha=120) -> Image.Image:
    base = img.convert("RGBA")
    H, W = mask01.shape
    if base.size != (W, H):
        base = base.resize((W, H), resample=Image.BILINEAR)
    m = (mask01.astype(np.uint8) * 255)
    maskL = Image.fromarray(m, mode="L")
    layer = Image.new("RGBA", base.size, (color[0], color[1], color[2], alpha))
    out = Image.composite(layer, base, maskL).convert("RGB")
    return out


def overlay_two_masks(img: Image.Image, gt01: np.ndarray, pred01: np.ndarray,
                      gt_color=(0, 255, 0), pred_color=(255, 0, 0),
                      gt_alpha=110, pred_alpha=110) -> Image.Image:
    out = overlay_mask(img, gt01, color=gt_color, alpha=gt_alpha)
    out = overlay_mask(out, pred01, color=pred_color, alpha=pred_alpha)
    return out


def error_map(img: Image.Image, gt01: np.ndarray, pred01: np.ndarray, alpha=140) -> Image.Image:
    """
    TP=green, FP=red, FN=blue (on top of original)
    """
    base = img.convert("RGBA")
    H, W = gt01.shape
    if base.size != (W, H):
        base = base.resize((W, H), resample=Image.BILINEAR)

    gt = gt01.astype(bool)
    pr = pred01.astype(bool)
    tp = np.logical_and(gt, pr).astype(np.uint8)
    fp = np.logical_and(~gt, pr).astype(np.uint8)
    fn = np.logical_and(gt, ~pr).astype(np.uint8)

    out = base
    out = Image.composite(Image.new("RGBA", base.size, (0, 255, 0, alpha)), out, Image.fromarray(tp * 255))
    out = Image.composite(Image.new("RGBA", base.size, (255, 0, 0, alpha)), out, Image.fromarray(fp * 255))
    out = Image.composite(Image.new("RGBA", base.size, (0, 0, 255, alpha)), out, Image.fromarray(fn * 255))
    return out.convert("RGB")


def make_4panel(img: Image.Image, gt_overlay: Image.Image, pred_overlay: Image.Image, both_overlay: Image.Image,
                title: str = "") -> Image.Image:
    imgs = [img.convert("RGB"), gt_overlay.convert("RGB"), pred_overlay.convert("RGB"), both_overlay.convert("RGB")]
    W = max(im.width for im in imgs)
    H = max(im.height for im in imgs)
    resized = [im.resize((W, H), resample=Image.BILINEAR) if im.size != (W, H) else im for im in imgs]
    canvas = Image.new("RGB", (W * 2, H * 2), (0, 0, 0))
    canvas.paste(resized[0], (0, 0))
    canvas.paste(resized[1], (W, 0))
    canvas.paste(resized[2], (0, H))
    canvas.paste(resized[3], (W, H))
    if title:
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, min(canvas.width, 1200), 28], fill=(0, 0, 0))
        d.text((8, 6), title[:160], fill=(255, 255, 255))
    return canvas


# =========================
# 5) MODELS: VLM / CLIP / SAM
# =========================
def _extract_json_obj(txt: str) -> Dict[str, Any]:
    s = txt.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    obj = s[start:end + 1]
    obj = re.sub(r",\s*}", "}", obj)
    obj = re.sub(r",\s*]", "]", obj)
    try:
        return json.loads(obj)
    except Exception:
        return {}


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
    def review_and_fix(self, review_img: Image.Image, expr: str, k: int,
                       cur_pos: List[str], cur_neg: List[str], max_new_tokens: int) -> Dict[str, Any]:
        prompt = f"""
You are a referring-segmentation reviewer and fixer.

You see one review image:
- Top: original image
- Bottom: K candidate tiles labeled idx=0..K-1, red area is the candidate mask.

Expression: {expr}

Current CLIP queries:
POS={cur_pos}
NEG={cur_neg}

Tasks:
1) If one candidate clearly matches the expression, output pick_idx (0..{k-1}). If none, set pick_idx=-1.
2) If CLIP queries are too long/ambiguous, rewrite them into SHORT English phrases:
   - pos: 2~6 short noun phrases describing the target (and relation if needed)
   - neg: 6~16 phrases to avoid common confusions (text/logo/background/person/hand etc.)
3) stop=true only if you are confident the chosen pick_idx is correct AND no further correction is needed.

Return ONLY strict JSON, no explanations:
{{"pick_idx":0,"stop":false,"pos":["a ..."],"neg":["text","logo"]}}
""".strip()

        self.counters.vlm_calls += 1

        if hasattr(self.proc, "apply_chat_template"):
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.proc(text=[text], images=[review_img], return_tensors="pt", padding=True)
        else:
            inputs = self.proc(text=[prompt], images=[review_img], return_tensors="pt", padding=True)

        if self.device == "cuda":
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        gen_ids = out[:, inputs["input_ids"].shape[1]:] if "input_ids" in inputs else out
        txt = self.proc.batch_decode(gen_ids, skip_special_tokens=True)[0]
        data = _extract_json_obj(txt)

        pick = data.get("pick_idx", 0)
        try:
            pick = int(pick)
        except Exception:
            pick = 0
        stop = bool(data.get("stop", False))

        pos = data.get("pos", None)
        neg = data.get("neg", None)
        if not isinstance(pos, list):
            pos = None
        if not isinstance(neg, list):
            neg = None

        return {"pick_idx": pick, "stop": stop, "pos": pos, "neg": neg, "raw": txt[:400]}


class ClipRanker:
    def __init__(self, clip_dir: str, counters: Counters):
        self.counters = counters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        p = Path(clip_dir)
        if not p.exists():
            raise FileNotFoundError(f"CLIP_DIR not found: {clip_dir}")

        has_st = any(p.glob("*.safetensors")) or (p / "model.safetensors").exists()
        has_bin = (p / "pytorch_model.bin").exists() or any(p.glob("pytorch_model*.bin"))
        if not has_st and not has_bin:
            raise OSError(f"No CLIP weights found in {clip_dir}. Files={ [x.name for x in p.glob('*')] }")

        self.model = CLIPModel.from_pretrained(
            clip_dir,
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=bool(has_st),
        ).to(self.device)
        self.proc = CLIPProcessor.from_pretrained(clip_dir, local_files_only=True)
        self.model.eval()

    @torch.inference_mode()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        self.counters.clip_text_calls += 1
        inp = self.proc(text=texts, images=None, return_tensors="pt", padding=True)
        inp = {k: v.to(self.device) for k, v in inp.items() if hasattr(v, "to")}
        feats = self.model.get_text_features(**inp)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats  # [T, d]

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
    def sims(img_feats: torch.Tensor, text_feats: torch.Tensor) -> np.ndarray:
        return (img_feats @ text_feats.T).detach().float().cpu().numpy()  # [N, T]


@dataclass
class Cand:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # xyxy
    pre: float  # predicted_iou + stability


def bbox_xywh_to_xyxy(b: List[float]) -> Tuple[int, int, int, int]:
    x, y, w, h = b
    return int(x), int(y), int(x + w), int(y + h)


def crop_bbox(img_rgb: np.ndarray, bbox: Tuple[int, int, int, int], pad: int) -> Image.Image:
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)
    return Image.fromarray(img_rgb[y1:y2 + 1, x1:x2 + 1])


def build_panel(img_rgb: np.ndarray, cands: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]],
                pad: int, tile: int = 224) -> Image.Image:
    K = len(cands)
    cols = 4 if K >= 8 else 2
    rows = int(np.ceil(K / cols))
    panel = Image.new("RGB", (cols * tile, rows * tile), (30, 30, 30))
    H, W = img_rgb.shape[:2]

    for i, (mask, bbox, score) in enumerate(cands):
        x1, y1, x2, y2 = bbox
        x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
        x2p = min(W - 1, x2 + pad); y2p = min(H - 1, y2 + pad)

        crop = img_rgb[y1p:y2p + 1, x1p:x2p + 1].copy()
        m = mask[y1p:y2p + 1, x1p:x2p + 1].astype(np.uint8) * 255

        base = Image.fromarray(crop).convert("RGBA")
        red = Image.new("RGBA", base.size, (255, 0, 0, 120))
        alpha = Image.fromarray(m).convert("L")
        over = Image.composite(red, base, alpha).convert("RGB")

        d = ImageDraw.Draw(over)
        d.rectangle([0, 0, 240, 28], fill=(0, 0, 0))
        d.text((8, 6), f"idx={i}  s={score:.3f}", fill=(255, 255, 255))

        over = over.resize((tile, tile), resample=Image.BILINEAR)
        r = i // cols
        cc = i % cols
        panel.paste(over, (cc * tile, r * tile))

    return panel


def stack_review(original: Image.Image, panel: Image.Image, max_w: int = 896) -> Image.Image:
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
# 6) RUN ONE SAMPLE (TRUE CLOSED-LOOP + VIS)
# =========================
def _default_pos_neg(expr: str, cfg: Cfg) -> Tuple[List[str], List[str]]:
    pos = []
    if cfg.init_pos_is_expr:
        pos.append(expr)

    neg = []
    if cfg.init_neg_bank:
        neg += ["text", "logo", "sign", "label", "caption",
                "background", "wall", "floor", "window", "door",
                "person", "face", "hair", "hand", "arm", "leg",
                "screen", "phone", "laptop"]

    pos = list(dict.fromkeys([p.strip() for p in pos if str(p).strip()]))
    neg = list(dict.fromkeys([n.strip() for n in neg if str(n).strip()]))
    if not pos:
        pos = [expr]
    return pos, neg


def _score_by_posneg(clip: ClipRanker, img_feats: torch.Tensor, pos: List[str], neg: List[str],
                     counters: Counters) -> np.ndarray:
    counters.clip_score_calls += 1

    pos = [str(x) for x in pos if str(x).strip()]
    neg = [str(x) for x in neg if str(x).strip()]
    if len(pos) == 0:
        pos = ["object"]
    if len(neg) == 0:
        neg = ["background"]

    pos_feats = clip.encode_texts(pos)
    neg_feats = clip.encode_texts(neg)

    sp = clip.sims(img_feats, pos_feats)  # [N, P]
    sn = clip.sims(img_feats, neg_feats)  # [N, Q]
    return sp.max(axis=1) - sn.max(axis=1)  # [N]


def run_one(expr: str, image_path: str, gt_mask: np.ndarray,
            clip: ClipRanker, vlm: Optional[VLM],
            amg, sam, SamPredictor,
            cfg: Cfg, counters: Counters,
            out_dir: Optional[Path] = None) -> Tuple[float, bool, int, int, Dict[str, float], Dict[str, Any]]:

    t0 = time.perf_counter()
    t_sam = t_clip = t_vlm = t_ref = 0.0

    img = Image.open(image_path).convert("RGB")
    img_rgb = np.array(img)
    H, W = img_rgb.shape[:2]

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # save raw original immediately
        img.save(out_dir / "img.jpg")

    # ---- A) SAM AMG once ----
    t_sam0 = time.perf_counter()
    counters.sam_amg_calls += 1
    amg_out = amg.generate(img_rgb)
    t_sam = time.perf_counter() - t_sam0

    cands: List[Cand] = []
    for d in amg_out:
        m = d["segmentation"].astype(np.uint8)
        af = float(m.sum()) / float(H * W)
        if af < cfg.min_area_frac or af > cfg.max_area_frac:
            continue
        bbox = bbox_xywh_to_xyxy(d.get("bbox", [0, 0, 0, 0]))
        pre = float(d.get("predicted_iou", 0.0) or 0.0) + float(d.get("stability_score", 0.0) or 0.0)
        cands.append(Cand(mask=m, bbox=bbox, pre=pre))

    if len(cands) == 0:
        times = {"t_sam": t_sam, "t_clip": 0.0, "t_vlm": 0.0, "t_refine": 0.0, "t_total": time.perf_counter() - t0}
        return 0.0, False, 0, 0, times, {"error": "No candidates"}

    cands.sort(key=lambda c: c.pre, reverse=True)
    cands = cands[:cfg.max_cands]

    # ---- B) CLIP image feats once (cache) ----
    t_clip0 = time.perf_counter()
    crops = [crop_bbox(img_rgb, c.bbox, cfg.bbox_pad) for c in cands]
    img_feats = clip.encode_images(crops, batch=cfg.clip_batch, use_amp=cfg.amp)
    t_clip_img = time.perf_counter() - t_clip0

    # ---- C) Outer loop: score -> topK -> VLM review&fix ----
    pos, neg = _default_pos_neg(expr, cfg)
    outer_used = 0
    chosen_global_idx = 0
    dbg_hist = []

    # write iter debug as jsonl
    iter_jsonl = None
    if out_dir is not None:
        iter_jsonl = open(out_dir / "iter_info.jsonl", "w", encoding="utf-8")

    for it in range(1, cfg.outer_max_iters + 1):
        outer_used = it

        # CLIP scoring (text-only)
        t_clip_it0 = time.perf_counter()
        scores = _score_by_posneg(clip, img_feats, pos, neg, counters)
        t_clip += (time.perf_counter() - t_clip_it0)

        idx_sorted = np.argsort(-scores)
        best = int(idx_sorted[0])
        second = int(idx_sorted[1]) if len(idx_sorted) > 1 else best
        gap = float(scores[best] - scores[second]) if len(idx_sorted) > 1 else 1e9

        K = min(cfg.topk_for_vlm, len(idx_sorted))
        top_idx = idx_sorted[:K].tolist()

        need_vlm = cfg.use_vlm and (vlm is not None) and (K > 1) and (cfg.force_vlm_each_iter or (gap < cfg.vlm_gap_trigger))

        pick_in_topk = 0
        stop = False
        pos_upd = None
        neg_upd = None
        raw_vlm = None

        review = None
        if need_vlm:
            top_for_panel = [(cands[j].mask, cands[j].bbox, float(scores[j])) for j in top_idx]
            panel = build_panel(img_rgb, top_for_panel, pad=cfg.bbox_pad, tile=cfg.vlm_tile)
            review = stack_review(img, panel, max_w=cfg.vlm_max_width)

            if out_dir is not None:
                review.save(out_dir / f"review_iter{it}.jpg")

            t_v0 = time.perf_counter()
            resp = vlm.review_and_fix(review, expr, k=K, cur_pos=pos, cur_neg=neg, max_new_tokens=cfg.vlm_max_new_tokens)
            t_vlm += (time.perf_counter() - t_v0)

            pick_in_topk = int(resp.get("pick_idx", 0))
            stop = bool(resp.get("stop", False))
            pos_upd = resp.get("pos", None)
            neg_upd = resp.get("neg", None)
            raw_vlm = resp.get("raw", None)

            if isinstance(pos_upd, list) and len(pos_upd) > 0:
                pos = list(dict.fromkeys([str(x).strip() for x in pos_upd if str(x).strip()]))[:6]
            if isinstance(neg_upd, list) and len(neg_upd) > 0:
                neg = list(dict.fromkeys([str(x).strip() for x in neg_upd if str(x).strip()]))[:16]

            if 0 <= pick_in_topk < K:
                chosen_global_idx = int(top_idx[pick_in_topk])
            else:
                chosen_global_idx = best
        else:
            chosen_global_idx = best

        # save per-iter chosen overlay
        if out_dir is not None:
            over_it = overlay_mask(img, cands[chosen_global_idx].mask, color=(255, 0, 0), alpha=120)
            over_it.save(out_dir / f"pred_iter{it}.jpg")

        dbg = {
            "iter": it,
            "need_vlm": bool(need_vlm),
            "gap": gap,
            "best": best,
            "second": second,
            "chosen_global_idx": int(chosen_global_idx),
            "chosen_score": float(scores[chosen_global_idx]),
            "clip_best_score": float(scores[best]),
            "pos": pos,
            "neg": neg,
            "pick_in_topk": int(pick_in_topk),
            "stop": bool(stop),
            "vlm_raw_head": raw_vlm,
        }
        dbg_hist.append(dbg)

        if iter_jsonl is not None:
            iter_jsonl.write(json.dumps(dbg, ensure_ascii=False) + "\n")
            iter_jsonl.flush()

        if stop:
            break

    if iter_jsonl is not None:
        iter_jsonl.close()

    # total CLIP time includes image-feats stage + text scoring stages
    t_clip = t_clip_img + t_clip

    chosen = cands[int(chosen_global_idx)]

    # ---- D) refine ----
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
    t_ref = time.perf_counter() - t_r0

    iou = mask_iou(refined, gt_mask)
    ok = (iou >= 0.5)

    # ---- E) VISUAL SAVE (final) ----
    if out_dir is not None:
        # GT
        _mask_to_pil(gt_mask).save(out_dir / "gt_mask.png")
        gt_over = overlay_mask(img, gt_mask, color=(0, 255, 0), alpha=120)
        gt_over.save(out_dir / "gt_overlay.jpg")

        # init chosen (before refine)
        _mask_to_pil(chosen.mask).save(out_dir / "init_mask.png")
        init_over = overlay_mask(img, chosen.mask, color=(255, 0, 0), alpha=120)
        init_over.save(out_dir / "init_overlay.jpg")

        # final pred (after refine)
        _mask_to_pil(refined).save(out_dir / "pred_mask.png")
        pred_over = overlay_mask(img, refined, color=(255, 0, 0), alpha=120)
        pred_over.save(out_dir / "pred_overlay.jpg")

        both = overlay_two_masks(img, gt_mask, refined)
        both.save(out_dir / "both_overlay.jpg")

        err = error_map(img, gt_mask, refined)
        err.save(out_dir / "error_map.jpg")

        title = f"{cfg.name} | IoU={iou:.3f} ok={ok} | outer={outer_used} refine={refine_iters} | expr={expr}"
        panel4 = make_4panel(img, gt_over, pred_over, both, title=title)
        panel4.save(out_dir / "compare_4panel.jpg")

        # save debug_loop.json for convenience
        (out_dir / "debug_loop.json").write_text(
            json.dumps({"outer_hist": dbg_hist, "outer_iters_used": outer_used, "refine_iters": refine_iters},
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    t_total = time.perf_counter() - t0
    times = {"t_sam": t_sam, "t_clip": t_clip, "t_vlm": t_vlm, "t_refine": t_ref, "t_total": t_total}

    debug_final = {
        "outer_hist": dbg_hist,
        "outer_iters_used": outer_used,
        "refine_iters": refine_iters,
    }
    return float(iou), bool(ok), int(outer_used), int(refine_iters), times, debug_final


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

    coco = COCO(INSTANCES_JSON)
    samples = load_refcoco_samples(REFS_PKL, coco, COCO_IMAGE_ROOTS, num_samples=cfg.num_samples, seed=cfg.seed)

    counters = Counters()

    print(f"\n[{cfg.name}] Loading models ...")
    clip = ClipRanker(CLIP_DIR, counters)
    vlm = VLM(VLM_DIR, counters) if cfg.use_vlm else None
    sam, amg, SamPredictor = load_sam_and_build_amg(SAM_CKPT, cfg)
    print(f"[{cfg.name}] Models loaded.")

    rows: List[OneResult] = []

    for i, s in enumerate(samples, 1):
        ref_id, image_id, ann_id, expr = s["ref_id"], s["image_id"], s["ann_id"], s["expr"]
        img_path = s["image_path"]

        ann = coco.loadAnns([ann_id])[0]
        gt_mask = coco.annToMask(ann).astype(np.uint8)

        snap0 = counters.snapshot()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        sample_out = out_dir / f"sample_{i:03d}"
        if cfg.save_debug_images:
            sample_out.mkdir(parents=True, exist_ok=True)

        iou, ok, outer_iters, refine_iters, times, dbg = run_one(
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
            out_dir=sample_out if cfg.save_debug_images else None,
        )

        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_resv = torch.cuda.max_memory_reserved() / (1024 ** 2)
        else:
            peak_alloc = 0.0
            peak_resv = 0.0

        snap1 = counters.snapshot()
        delta_calls = Counters.delta(snap1, snap0)

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
            tool_calls=delta_calls,
        )
        rows.append(row)

        print(f"[{cfg.name}] {i}/{len(samples)}  "
              f"IoU={iou:.3f} ok={ok} outer={outer_iters} refine={refine_iters}  "
              f"time={times['t_total']:.3f}s (sam={times['t_sam']:.3f}, clip={times['t_clip']:.3f}, "
              f"vlm={times['t_vlm']:.3f}, ref={times['t_refine']:.3f})  peakMB={peak_alloc:.1f}")

        with open(out_dir / "rows.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    summ = summarize(rows)
    pack = {"cfg": asdict(cfg), "summary": summ}
    (out_dir / "summary.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{cfg.name}] SUMMARY:\n{json.dumps(summ, ensure_ascii=False, indent=2)}")
    return summ, rows


def compare(base_sum: Dict[str, Any], fast_sum: Dict[str, Any]) -> Dict[str, Any]:
    def safe_ratio(a, b):
        return None if (a is None or b is None or b == 0) else float(a) / float(b)

    out = {
        "speedup_x": safe_ratio(base_sum.get("mean_time_s"), fast_sum.get("mean_time_s")),
        "mIoU_change": None if (base_sum.get("mIoU") is None or fast_sum.get("mIoU") is None) else float(fast_sum["mIoU"] - base_sum["mIoU"]),
        "success_change": None if (base_sum.get("success@0.5") is None or fast_sum.get("success@0.5") is None) else float(fast_sum["success@0.5"] - base_sum["success@0.5"]),
        "peak_alloc_mb_change": None if (base_sum.get("mean_peak_alloc_mb") is None or fast_sum.get("mean_peak_alloc_mb") is None) else float(fast_sum["mean_peak_alloc_mb"] - base_sum["mean_peak_alloc_mb"]),
        "tool_calls_change": {
            k: int(fast_sum["sum_tool_calls"].get(k, 0) - base_sum["sum_tool_calls"].get(k, 0))
            for k in base_sum.get("sum_tool_calls", {}).keys()
        }
    }
    return out


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

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
