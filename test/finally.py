import os
import re
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import CLIPModel, CLIPProcessor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

VLM_DIR   = r"E:\Vision1\vision"                 # Qwen3-VL 本地目录
CLIP_DIR  = r"E:\2026\CLIP_patch32"              # CLIP 本地目录
SAM_CKPT  = r"E:\2026\SAM1\sam_vit_b_01ec64.pth" # SAM ckpt
IMAGE_PATH = r"E:\Vision1\大模型领域.png"        # 测试图片
INSTRUCTION = "分割图片中中间的那个人的头"        # 指令
OUTDIR = r"E:\2026\3out"                         # 输出目录

MAX_ITERS = 3             # 至少 2 次就算闭环；建议 3
TOPK_FOR_VLM = 6          # 送给 VLM 看/选 的候选数量
SAM_POINTS_PER_SIDE = 32  # 候选数量/质量（越大越慢）
SAM_PRED_IOU_THRESH = 0.88
SAM_STABILITY_THRESH = 0.92

# =========================
# 数据结构
# =========================
@dataclass
class Candidate:
    local_id: int
    mask: np.ndarray  # HxW uint8 {0,1}
    bbox_xyxy: Tuple[int, int, int, int]
    area: int
    area_frac: float
    centroid_xy: Tuple[float, float]  # (cx, cy)
    pred_iou: float
    stability: float
    clip_score: float = float("-inf")


# =========================
# 工具：SAM
# =========================
def load_sam_predictor(sam_ckpt: str, device: str):
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except Exception as e:
        raise RuntimeError(
            "未能 import segment_anything。请先安装：pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from e

    # 从 ckpt 名字猜 vit_b / vit_l / vit_h
    ckpt_lower = os.path.basename(sam_ckpt).lower()
    if "vit_h" in ckpt_lower:
        model_type = "vit_h"
    elif "vit_l" in ckpt_lower:
        model_type = "vit_l"
    else:
        model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def generate_sam_candidates(image_rgb: np.ndarray, device: str) -> List[Candidate]:
    """
    用 SAM 自动生成候选 masks（主线1的“候选池”）。
    """
    try:
        from segment_anything import sam_model_registry
        from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
    except Exception as e:
        raise RuntimeError(
            "未能 import segment_anything.automatic_mask_generator。确认安装了 segment-anything。"
        ) from e

    # 选 vit_b / vit_l / vit_h（与 ckpt 一致）
    ckpt_lower = os.path.basename(SAM_CKPT).lower()
    if "vit_h" in ckpt_lower:
        model_type = "vit_h"
    elif "vit_l" in ckpt_lower:
        model_type = "vit_l"
    else:
        model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=SAM_CKPT)
    sam.to(device=device)

    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=SAM_POINTS_PER_SIDE,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_THRESH,
        box_nms_thresh=0.7,
        min_mask_region_area=0,
    )

    masks = gen.generate(image_rgb)  # list[dict]
    H, W = image_rgb.shape[:2]
    cands: List[Candidate] = []
    for i, m in enumerate(masks):
        seg = m["segmentation"].astype(np.uint8)  # HxW bool->uint8
        x, y, w, h = m["bbox"]  # xywh
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        area = int(seg.sum())
        if area <= 0:
            continue
        ys, xs = np.nonzero(seg)
        cx = float(xs.mean())
        cy = float(ys.mean())
        area_frac = area / float(H * W)
        pred_iou = float(m.get("predicted_iou", 0.0))
        stability = float(m.get("stability_score", 0.0))
        cands.append(
            Candidate(
                local_id=-1,
                mask=seg,
                bbox_xyxy=(x1, y1, x2, y2),
                area=area,
                area_frac=area_frac,
                centroid_xy=(cx, cy),
                pred_iou=pred_iou,
                stability=stability,
            )
        )
    return cands


# =========================
# 工具：CLIP 评分
# =========================
@torch.no_grad()
def clip_score_candidates(
    clip_model: CLIPModel,
    clip_proc: CLIPProcessor,
    image_pil: Image.Image,
    text: str,
    candidates: List[Candidate],
    device: str,
) -> List[Candidate]:
    """
    将 mask 区域做“抠图/裁剪”，用 CLIP 算 text-image 相似度。
    """
    # 文本 embedding
    text_inputs = clip_proc(text=[text], return_tensors="pt", padding=True).to(device)
    text_feat = clip_model.get_text_features(**text_inputs)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    img_np = np.array(image_pil.convert("RGB"))
    H, W = img_np.shape[:2]

    scored: List[Candidate] = []
    for c in candidates:
        x1, y1, x2, y2 = c.bbox_xyxy
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(1, min(x2, W))
        y2 = max(1, min(y2, H))

        crop = img_np[y1:y2, x1:x2].copy()
        mask_crop = c.mask[y1:y2, x1:x2].astype(bool)

        # 背景置灰：只保留 mask 区域纹理，有利于 CLIP 对齐“目标语义”
        bg = np.full_like(crop, 127)
        crop = np.where(mask_crop[..., None], crop, bg)

        crop_pil = Image.fromarray(crop)

        img_inputs = clip_proc(images=[crop_pil], return_tensors="pt").to(device)
        img_feat = clip_model.get_image_features(**img_inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        score = float((img_feat @ text_feat.T).squeeze().item())
        c.clip_score = score
        scored.append(c)

    scored.sort(key=lambda x: x.clip_score, reverse=True)
    for i, c in enumerate(scored):
        c.local_id = i  # 重新编号：0 是当前轮最高分
    return scored


# =========================
# 工具：可视化（回喂 VLM 用）
# =========================
def overlay_mask(img: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = img.convert("RGBA")
    H, W = mask.shape
    overlay = Image.new("RGBA", (W, H), (255, 0, 0, 0))
    ov_np = np.array(overlay)
    ov_np[mask.astype(bool)] = [255, 0, 0, int(255 * alpha)]
    overlay = Image.fromarray(ov_np, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    # 尽量用默认字体（Windows/Linux 都能跑）
    d.rectangle([0, 0, 220, 32], fill=(0, 0, 0))
    d.text((6, 6), text, fill=(255, 255, 255))
    return out


def make_board(original: Image.Image, topk: List[Candidate]) -> Image.Image:
    """
    拼一张图：左上是原图；其他格子是候选 overlay（带编号与CLIP分）。
    这样 VLM 只需要看“1张图”就能判断哪个候选更对，并提出纠错建议。
    """
    panels: List[Image.Image] = []
    panels.append(draw_label(original, "ORIG"))

    for c in topk:
        ov = overlay_mask(original, c.mask)
        x1, y1, x2, y2 = c.bbox_xyxy
        ov = draw_label(ov, f"ID={c.local_id} CLIP={c.clip_score:.3f} area={c.area_frac:.3f}")
        panels.append(ov)

    # 网格布局
    n = len(panels)
    cols = 3
    rows = math.ceil(n / cols)

    # 统一缩放到同尺寸（避免太大）
    W, H = original.size
    target_w = 420
    scale = target_w / float(W)
    target_h = int(H * scale)

    panels_rs = [p.resize((target_w, target_h), Image.BILINEAR) for p in panels]

    board = Image.new("RGB", (cols * target_w, rows * target_h), (30, 30, 30))
    for i, p in enumerate(panels_rs):
        r = i // cols
        c = i % cols
        board.paste(p, (c * target_w, r * target_h))
    return board


# =========================
# 工具：VLM 解析 + 纠错
# =========================
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型输出里扒出 JSON（模型可能会带多余文字）
    """
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        # 有时会单引号/多余逗号，做一次弱修复
        s2 = s.replace("'", '"')
        s2 = re.sub(r",\s*}", "}", s2)
        s2 = re.sub(r",\s*]", "]", s2)
        try:
            return json.loads(s2)
        except Exception:
            return None


@torch.no_grad()
def vlm_choose_and_refine(
    vlm_model: Qwen3VLForConditionalGeneration,
    vlm_proc: AutoProcessor,
    board_img: Image.Image,
    instruction: str,
    topk: List[Candidate],
    device: str,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """
    VLM 看拼接图 + 候选统计，选择 best_id，并给 refine_instruction（若需要下一轮）
    """
    # 给 VLM 的候选信息（文本辅助）
    lines = []
    for c in topk:
        x1, y1, x2, y2 = c.bbox_xyxy
        lines.append(
            f"- id={c.local_id}, clip={c.clip_score:.3f}, bbox=({x1},{y1},{x2},{y2}), area_frac={c.area_frac:.4f}"
        )
    cand_info = "\n".join(lines)

    prompt = f"""
你在一张拼接图里看到：
- 左上角 ORIG：原图
- 其它格子：候选分割结果（红色区域），每个格子左上角写着 ID、CLIP 分数、area_frac

用户指令：{instruction}

候选列表（文字版）：
{cand_info}

请你做两件事：
1) 如果某个候选明显满足“用户指令”，请选择最合适的 best_id。
2) 如果都不满意或歧义很大，请给出下一轮更具体的 refine_instruction（要更可定位，例如加入方位、部位、排除项）。

必须输出【严格JSON】（不要输出其它内容）：
{{
  "ok": true/false,
  "best_id": 0,
  "refine_instruction": "……",
  "negative": ["……", "……"],
  "reason": "简短说明"
}}
其中：
- ok=true 表示你认为 best_id 已经很符合指令
- ok=false 表示需要下一轮 refine_instruction
"""

    # Qwen3-VL 常见用法：processor 直接吃 image + text
    inputs = vlm_proc(images=board_img, text=prompt, return_tensors="pt").to(device)
    out = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = vlm_proc.batch_decode(out, skip_special_tokens=True)[0]

    js = extract_json(decoded)
    if js is None:
        # 兜底：如果模型没按 JSON 来，就强制进入下一轮
        js = {"ok": False, "best_id": 0, "refine_instruction": instruction, "negative": [], "reason": "VLM未返回可解析JSON"}
    return js


# =========================
# 主循环：闭环 Ref-Seg Loop
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_mask(mask: np.ndarray, path: str):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def main():
    ensure_dir(OUTDIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)

    # 读图
    img = Image.open(IMAGE_PATH).convert("RGB")
    img_np = np.array(img)  # HWC RGB
    H, W = img_np.shape[:2]

    # 加载 CLIP
    print("[INFO] load CLIP:", CLIP_DIR)
    clip_model = CLIPModel.from_pretrained(CLIP_DIR).to(device)
    clip_proc = CLIPProcessor.from_pretrained(CLIP_DIR)

    # 加载 VLM
    print("[INFO] load VLM:", VLM_DIR)
    vlm_proc = AutoProcessor.from_pretrained(VLM_DIR)
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        vlm_model = vlm_model.to(device)

    instruction = INSTRUCTION
    final_mask = None
    final_info = None

    for it in range(1, MAX_ITERS + 1):
        print(f"\n[LOOP] Iteration {it}/{MAX_ITERS}")
        print("[LOOP] instruction:", instruction)

        # 1) SAM 生成候选
        cands = generate_sam_candidates(img_np, device=device)
        print("[SAM] raw candidates:", len(cands))
        if len(cands) == 0:
            print("[ERROR] SAM 没生成任何候选。")
            return

        # 2) CLIP 对候选打分（验证器）
        scored = clip_score_candidates(
            clip_model=clip_model,
            clip_proc=clip_proc,
            image_pil=img,
            text=instruction,
            candidates=cands,
            device=device,
        )

        topk = scored[:TOPK_FOR_VLM]
        print("[CLIP] top1 score:", topk[0].clip_score)

        # 3) 回喂给 VLM：看拼接图 + 选择 best_id / 给 refine
        board = make_board(img, topk)
        board_path = os.path.join(OUTDIR, f"iter{it}_board.jpg")
        board.save(board_path)
        print("[VLM] board saved:", board_path)

        js = vlm_choose_and_refine(
            vlm_model=vlm_model,
            vlm_proc=vlm_proc,
            board_img=board,
            instruction=instruction,
            topk=topk,
            device=device,
        )
        print("[VLM] json:", js)

        ok = bool(js.get("ok", False))
        best_id = int(js.get("best_id", 0))
        refine = str(js.get("refine_instruction", instruction)).strip()

        # 找到 best_id 对应的 mask（best_id 是 topk 里的 local_id）
        chosen = None
        for c in topk:
            if c.local_id == best_id:
                chosen = c
                break
        if chosen is None:
            chosen = topk[0]

        # 保存本轮选择结果
        ov = overlay_mask(img, chosen.mask)
        ov = draw_label(ov, f"CHOSEN it{it} id={chosen.local_id} clip={chosen.clip_score:.3f}")
        ov_path = os.path.join(OUTDIR, f"iter{it}_chosen_overlay.jpg")
        ov.save(ov_path)

        mask_path = os.path.join(OUTDIR, f"iter{it}_chosen_mask.png")
        save_mask(chosen.mask, mask_path)

        final_mask = chosen.mask
        final_info = {
            "iter": it,
            "instruction": instruction,
            "chosen_id": chosen.local_id,
            "clip": chosen.clip_score,
            "bbox_xyxy": chosen.bbox_xyxy,
            "area_frac": chosen.area_frac,
            "vlm": js,
        }

        # 4) 是否停止；否则用 VLM 的 refine_instruction 再来一轮（闭环）
        if ok:
            print("[STOP] VLM认为已满足指令，停止闭环。")
            break

        # 如果 VLM 没给更具体的，就轻微加强：追加“更具体描述”
        if len(refine) < 3:
            refine = instruction + "，请更精确到目标部位/方位并排除相似物体"

        instruction = refine

    # 最终输出
    if final_mask is not None:
        json_path = os.path.join(OUTDIR, "final_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_info, f, ensure_ascii=False, indent=2)
        print("\n[DONE] final saved:", json_path)
        print("[DONE] check overlays/masks in:", OUTDIR)


if __name__ == "__main__":
    main()
