"""
ArcFace (insightface) cosine similarity between a GT face image and frames of a same-basename video.
Unlike a variant that skips no-face frames: if a frame has no face, similarity is 0 and still counts
toward the video mean (penalizes missing faces).
- Read GT images from folder A and videos from folder B (paired by basename).
- Per video: per-frame sim = cosine(GT, frame) if face else 0; video mean = id_csim.
- Mean over videos, append row to CSV.
"""

import argparse
import csv
import os
from pathlib import Path

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


def get_arcface_embedding_and_crop(
    app: FaceAnalysis, img_bgr: np.ndarray, crop_region=None
):
    """
    One ArcFace embedding and face crop from BGR image (insightface FaceAnalysis).
    Returns (None, None) if no face; if multiple faces, largest bbox wins.
    """
    if crop_region is not None:
        x, y, w, h = crop_region
        img_bgr = crop_image(img_bgr, x, y, w, h)

    try:
        faces = app.get(img_bgr)
    except Exception:
        return None, None
    if not faces:
        return None, None

    areas = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        areas.append((x2 - x1) * (y2 - y1))
    best_idx = int(np.argmax(areas))
    face = faces[best_idx]
    emb = face.normed_embedding

    x1, y1, x2, y2 = face.bbox
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
    face_crop = img_bgr[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None

    return emb.astype(np.float32), face_crop


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


def main():
    parser = argparse.ArgumentParser(
        description="ArcFace id similarity: no-face frames count as csim=0 in the mean"
    )
    parser.add_argument("--gt_dir", "-a", required=True, help="Folder A: GT face images")
    parser.add_argument("--video_dir", "-b", required=True, help="Folder B: generated videos (same basename as GT)")
    parser.add_argument("--output", "-o", default="./id_csim_arcface_with_zeros.csv", help="Output CSV path")
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        default=None,
        metavar=("X", "Y", "W", "H"),
        help="Crop region before detection; omit for full frame",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None, help="Max frames per video (default: all)"
    )
    parser.add_argument(
        "--sample_frames",
        type=int,
        default=None,
        help="Uniformly sample this many frames (alternative to max_frames)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=0,
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
        help="Optional: save cropped GT face and first matched frame face here",
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

    per_video_csim = []

    for gt_path, video_path in tqdm(pairs, desc="videos"):
        gt_img = cv2.imread(gt_path)
        if gt_img is None:
            tqdm.write(f"Skip invalid GT image: {gt_path}")
            continue
        gt_emb, gt_face_crop = get_arcface_embedding_and_crop(
            app, gt_img, crop_region=crop_region
        )
        if gt_emb is None:
            tqdm.write(f"No face in GT, skip: {gt_path}")
            continue

        stem = Path(gt_path).stem
        if save_faces_dir is not None and gt_face_crop is not None:
            cv2.imwrite(str(save_faces_dir / f"{stem}_gt.png"), gt_face_crop)

        raw_frames = get_video_frames_bgr(
            video_path, crop_region=crop_region, max_frames=args.max_frames
        )
        if not raw_frames:
            tqdm.write(f"Skip empty video: {video_path}")
            continue

        if args.sample_frames is not None and len(raw_frames) > args.sample_frames:
            indices = np.linspace(0, len(raw_frames) - 1, args.sample_frames, dtype=int)
            raw_frames = [raw_frames[i] for i in indices]

        # Every frame counts: face -> sim, no face -> 0
        frame_sims = []
        saved_frame_crop = False
        for frame_bgr in raw_frames:
            frame_emb, frame_face_crop = get_arcface_embedding_and_crop(
                app, frame_bgr, crop_region=None
            )
            if frame_emb is None:
                frame_sims.append(0.0)
                continue
            sim = float(np.dot(gt_emb, frame_emb))
            frame_sims.append(sim)

            if (
                save_faces_dir is not None
                and (not saved_frame_crop)
                and frame_face_crop is not None
            ):
                cv2.imwrite(str(save_faces_dir / f"{stem}_frame.png"), frame_face_crop)
                saved_frame_crop = True

        # Mean includes zeros; do not skip video if all frames lack faces
        video_csim = float(np.mean(frame_sims))
        per_video_csim.append(video_csim)

    if not per_video_csim:
        print("No video produced a valid id_csim (ArcFace).")
        return

    id_csim = sum(per_video_csim) / len(per_video_csim)
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["metric", "overall_mean", "person0_mean", "person1_mean", "num_videos"])
        w.writerow(
            ["id_csim_single", f"{id_csim:.6f}", "", "", len(per_video_csim)],
        )
    print(
        f"Processed {len(per_video_csim)} videos (no-face frames = 0); "
        f"ArcFace id_csim = {id_csim:.6f}; appended to {args.output}"
    )


if __name__ == "__main__":
    main()
