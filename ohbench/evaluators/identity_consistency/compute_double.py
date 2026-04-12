"""
ArcFace (insightface) two-person ID similarity between a GT image and same-basename video.
- For two-person-in-frame videos: compute id_csim per person, then average the two.
- If a frame lacks a matched face for someone, that person's sim is 0 for that frame.

Pipeline:
- GT images from folder A (one image with two main faces per clip)
- Videos from folder B (same basename)
- Per (GT, video):
  - Detect two largest faces on GT, sort by bbox x1 ascending -> person0 / person1
  - Per frame: detect up to two faces (largest areas); match to GT embeddings
    - Two faces: pick assignment with higher total similarity
    - One face: assign to closer GT embedding, other person 0
    - Zero faces: both 0
  - Per-person mean over frames -> video_id_csim per person; average -> video_avg
- Append summary CSV
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from insightface.app import FaceAnalysis

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def crop_image(image, x, y, width, height):
    return image[y : y + height, x : x + width]


def find_paired_files(gt_dir, video_dir):
    """
    Pair by basename: GT stem == video stem (no extension).
    Returns [(gt_path, video_path), ...]
    """
    gt_paths = {}
    for f in os.listdir(gt_dir):
        p = Path(f)
        if p.suffix.lower() in IMG_EXTENSIONS:
            gt_paths[p.stem] = os.path.join(gt_dir, f)

    pairs = []
    for f in os.listdir(video_dir):
        p = Path(f)
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            stem = p.stem
            if stem in gt_paths:
                pairs.append((gt_paths[stem], os.path.join(video_dir, f)))
    return pairs


def _select_top_faces_by_area(faces, max_faces: int = 2):
    if not faces:
        return []
    areas = []
    for idx, f in enumerate(faces):
        x1, y1, x2, y2 = f.bbox
        areas.append(((x2 - x1) * (y2 - y1), idx))
    areas.sort(reverse=True, key=lambda t: float(t[0]))
    return [faces[idx] for _area, idx in areas[:max_faces]]


def _bbox_int_xyxy(bbox, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return x1, y1, x2, y2


def get_arcface_two_embeddings_and_crops(
    app: FaceAnalysis, img_bgr: np.ndarray, crop_region=None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Two largest faces on BGR image -> emb0, emb1, crop0, crop1.
    Returns (None, None, None, None) if fewer than two faces.

    person0/person1: sorted by face bbox x1 ascending (left = person0).
    """
    if crop_region is not None:
        x, y, w, h = crop_region
        img_bgr = crop_image(img_bgr, x, y, w, h)

    try:
        faces = app.get(img_bgr)
    except Exception:
        return None, None, None, None

    top = _select_top_faces_by_area(faces, max_faces=2)
    if len(top) < 2:
        return None, None, None, None

    # left-to-right ordering for stable identity slots
    top.sort(key=lambda f: float(f.bbox[0]))

    embs: List[np.ndarray] = []
    crops: List[np.ndarray] = []
    for f in top:
        emb = f.normed_embedding.astype(np.float32)
        x1, y1, x2, y2 = _bbox_int_xyxy(f.bbox, img_bgr.shape[1], img_bgr.shape[0])
        crop = img_bgr[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None
        embs.append(emb)
        crops.append(crop)

    if crops[0] is None or crops[1] is None:
        return None, None, None, None
    return embs[0], embs[1], crops[0], crops[1]


def get_video_frames_bgr(video_path, crop_region=None, max_frames=None):
    """Read video; return list of BGR frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if crop_region is not None:
            x, y, w, h = crop_region
            frame = crop_image(frame, x, y, w, h)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def _frame_match_sims(
    gt0: np.ndarray, gt1: np.ndarray, frame_faces: List
) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Given GT embeddings and detected faces (top-2) for one frame:
      sim0, sim1, crop_for_p0, crop_for_p1
    Rules:
    - 2 faces: pick assignment with higher total similarity
    - 1 face: assign to closer GT, other 0
    - 0 faces: both 0
    """
    if not frame_faces:
        return 0.0, 0.0, None, None

    # keep at most 2
    frame_faces = _select_top_faces_by_area(frame_faces, max_faces=2)

    def face_emb_and_crop(face):
        emb = face.normed_embedding.astype(np.float32)
        x1, y1, x2, y2 = _bbox_int_xyxy(face.bbox, _W, _H)
        crop = _IMG[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None
        return emb, crop

    # use closure vars set by caller
    if len(frame_faces) == 1:
        emb, crop = face_emb_and_crop(frame_faces[0])
        s0 = float(np.dot(gt0, emb))
        s1 = float(np.dot(gt1, emb))
        if s0 >= s1:
            return s0, 0.0, crop, None
        return 0.0, s1, None, crop

    emb_a, crop_a = face_emb_and_crop(frame_faces[0])
    emb_b, crop_b = face_emb_and_crop(frame_faces[1])

    s00 = float(np.dot(gt0, emb_a))
    s01 = float(np.dot(gt0, emb_b))
    s10 = float(np.dot(gt1, emb_a))
    s11 = float(np.dot(gt1, emb_b))

    # assignment 1: gt0->a, gt1->b ; assignment 2: gt0->b, gt1->a
    if (s00 + s11) >= (s01 + s10):
        return s00, s11, crop_a, crop_b
    return s01, s10, crop_b, crop_a


def _append_summary_row(
    out_csv: Path,
    metric: str,
    overall_mean: float,
    person0_mean: float,
    person1_mean: float,
    num_videos: int,
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["metric", "overall_mean", "person0_mean", "person1_mean", "num_videos"])
        w.writerow([metric, f"{overall_mean:.6f}", f"{person0_mean:.6f}", f"{person1_mean:.6f}", num_videos])


def main():
    parser = argparse.ArgumentParser(
        description="ArcFace two-person id similarity: missing-face frames count as 0 per person"
    )
    parser.add_argument("--gt_dir", "-a", required=True, help="GT images dir (two faces each)")
    parser.add_argument("--video_dir", "-b", required=True, help="Generated videos dir (same basename as GT)")
    parser.add_argument(
        "--output",
        "-o",
        default="./id_csim_double/id_csim_arcface_two_people_with_zeros_uniavgen.csv",
        help="Summary CSV path (append)",
    )
    parser.add_argument(
        "--per_video_csv",
        type=str,
        default=None,
        help="Optional: per-video person0/person1/avg CSV",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        default=None,
        metavar=("X", "Y", "W", "H"),
        help="Crop region before detection; omit for full frame",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames per video (default: all)")
    parser.add_argument(
        "--sample_frames",
        type=int,
        default=None,
        help="Uniformly sample this many frames (alternative to max_frames)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=2,
        help="insightface ctx id: 0 = GPU0, -1 = CPU",
    )
    parser.add_argument(
        "--det_size",
        nargs=2,
        type=int,
        default=[640, 640],
        metavar=("W", "H"),
        help="Face detector input size, default 640x640",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="buffalo_l",
        help="insightface FaceAnalysis model name, default buffalo_l",
    )
    parser.add_argument(
        "--save_faces_dir",
        type=str,
        default=None,
        help="Optional: save GT two crops and first matched frame crops",
    )
    args = parser.parse_args()

    crop_region = tuple(args.crop) if args.crop is not None else None
    save_faces_dir = Path(args.save_faces_dir) if args.save_faces_dir else None
    if save_faces_dir is not None:
        save_faces_dir.mkdir(parents=True, exist_ok=True)
        print(f"Crops will be saved to: {save_faces_dir.resolve()}")

    app = FaceAnalysis(name=args.model)
    app.prepare(ctx_id=args.ctx, det_size=tuple(args.det_size))

    pairs = find_paired_files(args.gt_dir, args.video_dir)
    if not pairs:
        print(
            f"No basename pairs between {args.gt_dir} and {args.video_dir} "
            f"(images: {IMG_EXTENSIONS}, videos: {VIDEO_EXTENSIONS})"
        )
        return

    per_video_avg: List[float] = []
    per_video_p0: List[float] = []
    per_video_p1: List[float] = []

    per_video_rows = []

    print(f"{len(pairs)} (GT, video) pairs; per-video:")
    for gt_path, video_path in tqdm(pairs, desc="videos"):
        gt_img = cv2.imread(gt_path)
        if gt_img is None:
            tqdm.write(f"Skip invalid GT image: {gt_path}")
            continue

        gt0, gt1, gt_crop0, gt_crop1 = get_arcface_two_embeddings_and_crops(
            app, gt_img, crop_region=crop_region
        )
        if gt0 is None or gt1 is None:
            tqdm.write(f"GT needs two faces, skip: {gt_path}")
            continue

        stem = Path(gt_path).stem
        if save_faces_dir is not None:
            cv2.imwrite(str(save_faces_dir / f"{stem}_gt_p0.png"), gt_crop0)
            cv2.imwrite(str(save_faces_dir / f"{stem}_gt_p1.png"), gt_crop1)

        raw_frames = get_video_frames_bgr(
            video_path, crop_region=crop_region, max_frames=args.max_frames
        )
        if not raw_frames:
            tqdm.write(f"Skip empty video: {video_path}")
            continue

        if args.sample_frames is not None and len(raw_frames) > args.sample_frames:
            indices = np.linspace(0, len(raw_frames) - 1, args.sample_frames, dtype=int)
            raw_frames = [raw_frames[i] for i in indices]

        sims0: List[float] = []
        sims1: List[float] = []
        saved_p0 = False
        saved_p1 = False

        for frame_bgr in raw_frames:
            global _IMG, _H, _W  # used by _frame_match_sims crop helper
            _IMG = frame_bgr
            _H, _W = frame_bgr.shape[0], frame_bgr.shape[1]

            try:
                faces = app.get(frame_bgr)
            except Exception:
                faces = []

            s0, s1, crop_p0, crop_p1 = _frame_match_sims(gt0, gt1, faces)
            sims0.append(float(s0))
            sims1.append(float(s1))

            if save_faces_dir is not None:
                if (not saved_p0) and crop_p0 is not None:
                    cv2.imwrite(str(save_faces_dir / f"{stem}_frame_p0.png"), crop_p0)
                    saved_p0 = True
                if (not saved_p1) and crop_p1 is not None:
                    cv2.imwrite(str(save_faces_dir / f"{stem}_frame_p1.png"), crop_p1)
                    saved_p1 = True

        video_p0 = float(np.mean(sims0)) if sims0 else 0.0
        video_p1 = float(np.mean(sims1)) if sims1 else 0.0
        video_avg = float((video_p0 + video_p1) / 2.0)

        per_video_p0.append(video_p0)
        per_video_p1.append(video_p1)
        per_video_avg.append(video_avg)

        tqdm.write(f"  {stem}: person0={video_p0:.6f}, person1={video_p1:.6f}, avg={video_avg:.6f}")

        if args.per_video_csv is not None:
            per_video_rows.append([stem, f"{video_p0:.6f}", f"{video_p1:.6f}", f"{video_avg:.6f}"])

    if not per_video_avg:
        print("No video produced two-person id_csim (ArcFace).")
        return

    overall_mean = float(np.mean(per_video_avg))
    person0_mean = float(np.mean(per_video_p0))
    person1_mean = float(np.mean(per_video_p1))

    _append_summary_row(
        out_csv=Path(args.output),
        metric="id_csim_double",
        overall_mean=overall_mean,
        person0_mean=person0_mean,
        person1_mean=person1_mean,
        num_videos=len(per_video_avg),
    )

    if args.per_video_csv is not None:
        p = Path(args.per_video_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "person0", "person1", "avg"])
            w.writerows(per_video_rows)
        print(f"Per-video CSV: {p}")

    print(
        f"\nProcessed {len(per_video_avg)} videos (missing faces = 0 per frame)."
    )
    print(
        f"Means: person0_mean={person0_mean:.6f}, person1_mean={person1_mean:.6f}, "
        f"overall_avg={overall_mean:.6f}"
    )
    print(f"Appended to {args.output}")


if __name__ == "__main__":
    main()
