from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Detection dataclass
@dataclass
class Detection:
    frame_id: int
    bbox: tuple          #(x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    crop: np.ndarray = field(repr=False)  #BGR crop 

    @property
    def foot_point(self) -> tuple[float, float]:
        """Input for Homography (Pitch-plane tracking)"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))


# ════════════════════════════════════════════════════
# Helpers
def pick_device(requested: str) -> str:
    """Auto-detect GPU nếu requested='auto'; validate nếu chỉ định thủ công."""
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            print("⚠️  Yêu cầu CUDA nhưng không có GPU khả dụng -> fallback CPU.")
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def run_detection(
    model: YOLO,
    video_path: Path,
    out_video_path: Path,
    cfg: dict,
    video_info: dict,
    max_frames: int | None = None,
) -> dict[int, list[Detection]]:
    width, height, fps = video_info["width"], video_info["height"], video_info["fps"]
    total = max_frames or video_info["total_frames"]

    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(
        str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    all_detections: dict[int, list[Detection]] = {}
    frame_id = 0
    pbar = tqdm(total=total, desc="Detecting", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_id >= max_frames):
            break

        results = model.predict(
            source=frame,
            conf=cfg["conf"],
            iou=cfg["iou"],
            imgsz=cfg["imgsz"],
            half=cfg["half"],
            device=cfg["device"],
            classes=cfg["classes"],
            verbose=False,
        )[0]

        frame_dets: list[Detection] = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                pad = 10
                crop = frame[
                    max(0, y1 - pad): min(height, y2 + pad),
                    max(0, x1 - pad): min(width, x2 + pad),
                ].copy()
                frame_dets.append(
                    Detection(
                        frame_id=frame_id,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=model.names[cls_id],
                        crop=crop,
                    )
                )

        all_detections[frame_id] = frame_dets

        vis = frame.copy()
        for det in frame_dets:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                vis, f"{det.class_name} {det.confidence:.2f}", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA,
            )
            fx, fy = det.foot_point
            cv2.circle(vis, (int(fx), int(fy)), 4, (0, 255, 0), -1)

        cv2.putText(
            vis, f"Frame {frame_id} | Dets: {len(frame_dets)}", (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(vis)

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    return all_detections


def save_sample_frame(video_path: Path, all_detections: dict, sample_id: int, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_id)
    ok, sample = cap.read()
    cap.release()
    if not ok:
        print(f"⚠️  Không đọc được frame {sample_id} để lưu preview.")
        return

    for det in all_detections.get(sample_id, []):
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(sample, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            sample, f"{det.class_name} {det.confidence:.2f}", (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
        )
        fx, fy = det.foot_point
        cv2.circle(sample, (int(fx), int(fy)), 5, (0, 255, 0), -1)

    cv2.imwrite(str(out_path), sample)
    print(f"🖼️  Preview frame {sample_id} -> {out_path}")


#Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1 - Player detection (local PC / Jetson-ready)")
    parser.add_argument("--model", required=True, type=Path, help="Đường dẫn file .pt (best.pt)")
    parser.add_argument("--video", required=True, type=Path, help="Video đầu vào")
    parser.add_argument("--outdir", default=Path("./output"), type=Path, help="Thư mục lưu kết quả")
    parser.add_argument("--conf", default=0.7, type=float)
    parser.add_argument("--iou", default=0.45, type=float)
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--no-half", dest="half", action="store_false", default=True,
                     help="Tắt FP16 (mặc định bật nếu chạy GPU)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--classes", nargs="*", type=int, default=None,
                         help="Danh sách class_id cần giữ lại (index theo model.names). Bỏ trống = lấy tất cả.")
    parser.add_argument("--max-frames", default=None, type=int, help="Giới hạn số frame để test nhanh")
    parser.add_argument("--sample-frame", default=50, type=int, help="Frame id để xuất ảnh preview")
    args = parser.parse_args()

    if not args.model.exists():
        sys.exit(f"❌ Không tìm thấy model: {args.model}")
    if not args.video.exists():
        sys.exit(f"❌ Không tìm thấy video: {args.video}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_video = args.outdir / "step1_output.mp4"
    out_pkl = args.outdir / "step1_detections.pkl"
    out_preview = args.outdir / "step1_sample_frame.jpg"

    device = pick_device(args.device)
    half = args.half and device == "cuda"  # FP16 không có tác dụng / không ổn định trên CPU

    print(f"🖥️  Device: {device} | half={half}")
    print(f"📦 Loading model: {args.model}")
    model = YOLO(str(args.model))
    model.to(device)

    print("📋 Class names trong model:")
    for idx, name in model.names.items():
        print(f"   [{idx}] {name}")

    video_info = get_video_info(args.video)
    duration_min = video_info["total_frames"] / video_info["fps"] / 60
    print(f"📹 {video_info['width']}x{video_info['height']} | "
          f"{video_info['fps']:.1f} fps | {video_info['total_frames']} frames | "
          f"{duration_min:.1f} phút")

    cfg = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "half": half,
        "device": device,
        "classes": args.classes,  # None = tất cả class
    }
    print(f"✅ Config: {cfg}")

    t0 = time.time()
    all_detections = run_detection(
        model, args.video, out_video, cfg, video_info, max_frames=args.max_frames
    )
    elapsed = time.time() - t0

    total_dets = sum(len(v) for v in all_detections.values())
    n_frames = len(all_detections)
    print("✅ Done!")
    print(f"   Frames processed : {n_frames}")
    print(f"   Total detections  : {total_dets}")
    print(f"   Time/frame : {total_dets / max(n_frames, 1):.1f}")
    print(f"   Total time  : {elapsed:.1f}s  ({n_frames / max(elapsed, 1e-6):.2f} FPS)")
    print(f"   Video output     : {out_video}")

    with open(out_pkl, "wb") as f:
        pickle.dump(all_detections, f)
    print(f"✅ Đã lưu detections -> {out_pkl}")
    print(f'   Bước 2 load bằng: all_detections = pickle.load(open("{out_pkl}", "rb"))')

    save_sample_frame(args.video, all_detections, args.sample_frame, out_preview)


if __name__ == "__main__":
    main()
