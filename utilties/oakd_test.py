#!/usr/bin/env python3
import os
from pathlib import Path

for _font_dir in (
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation2"),
    Path("/usr/share/fonts/truetype/freefont"),
):
    if _font_dir.exists():
        os.environ.setdefault("QT_QPA_FONTDIR", str(_font_dir))
        break

import cv2
import depthai as dai
import time

MODEL_NAME = "luxonis/yolov8-nano-pose-estimation:coco-512x288"

# Fallback body edges for common 17/18-keypoint human-pose layouts.
BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _to_xy(point) -> tuple[int, int] | None:
    if point is None:
        return None
    if hasattr(point, "x") and hasattr(point, "y"):
        return int(round(float(point.x))), int(round(float(point.y)))
    if hasattr(point, "imageCoordinates"):
        img_pt = point.imageCoordinates
        if hasattr(img_pt, "x") and hasattr(img_pt, "y"):
            return int(round(float(img_pt.x))), int(round(float(img_pt.y)))
    return None


def _extract_instances(parsed_msg):
    edges = BODY_EDGES
    if hasattr(parsed_msg, "getEdges"):
        try:
            raw_edges = parsed_msg.getEdges()
            if raw_edges:
                parsed_edges = []
                for edge in raw_edges:
                    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                        parsed_edges.append((int(edge[0]), int(edge[1])))
                    elif hasattr(edge, "x") and hasattr(edge, "y"):
                        parsed_edges.append((int(edge.x), int(edge.y)))
                if parsed_edges:
                    edges = parsed_edges
        except Exception:
            pass

    # Newer parser outputs may expose detections with keypoints attached.
    if hasattr(parsed_msg, "detections"):
        for det in parsed_msg.detections:
            keypoints = []
            if hasattr(det, "getKeypoints2f"):
                try:
                    for pt in det.getKeypoints2f():
                        xy = _to_xy(pt)
                        keypoints.append(xy)
                except Exception:
                    pass
            elif hasattr(det, "getKeypoints"):
                try:
                    for pt in det.getKeypoints():
                        keypoints.append(_to_xy(pt))
                except Exception:
                    pass
            if keypoints:
                yield keypoints, edges
        return

    # Some parser outputs may expose only a keypoints list.
    if hasattr(parsed_msg, "getKeypoints"):
        try:
            groups = parsed_msg.getKeypoints()
        except Exception:
            groups = []
        for group in groups or []:
            keypoints = []
            for pt in group:
                keypoints.append(_to_xy(pt))
            if keypoints:
                yield keypoints, edges


def draw_skeleton(frame, parsed_msg) -> None:
    for keypoints, edges in _extract_instances(parsed_msg):
        for idx, xy in enumerate(keypoints):
            if xy is None:
                continue
            cv2.circle(frame, xy, 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                str(idx),
                (xy[0] + 4, xy[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

        for a, b in edges:
            if a >= len(keypoints) or b >= len(keypoints):
                continue
            p1 = keypoints[a]
            p2 = keypoints[b]
            if p1 is None or p2 is None:
                continue
            cv2.line(frame, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    with dai.Pipeline() as pipeline:
        camera = pipeline.create(dai.node.Camera).build()
        nn = pipeline.create(dai.node.NeuralNetwork).build(
            camera,
            dai.NNModelDescription(MODEL_NAME),
        )
        parser = pipeline.create(dai.node.DetectionParser).build(
            nn.out,
            nn.getNNArchive(),
        )
        try:
            parser.setDecodeKeypoints(True)
        except Exception:
            pass

        q_rgb = nn.passthrough.createOutputQueue()
        q_pose = parser.out.createOutputQueue()

        pipeline.start()

        device = pipeline.getDefaultDevice()
        print("Device:", device.getDeviceInfo().getDeviceId())
        print("USB speed:", device.getUsbSpeed())
        print("Cameras:", device.getConnectedCameras())
        print(f"Running DepthAI v3 pose model: {MODEL_NAME}")
        print("Press q to quit")

        start_time = time.monotonic()
        counter = 0

        while pipeline.isRunning():
            in_rgb = q_rgb.get()
            in_pose = q_pose.get()

            frame = in_rgb.getCvFrame()
            counter += 1

            draw_skeleton(frame, in_pose)

            fps = counter / max(time.monotonic() - start_time, 1e-6)
            cv2.putText(
                frame,
                f"Pose FPS: {fps:.2f}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("OAK-D v3 Skeleton Tracking", frame)
            if cv2.waitKey(1) == ord("q"):
                pipeline.stop()
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
