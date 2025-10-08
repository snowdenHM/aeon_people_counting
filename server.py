import os
import json
import time
import uuid
import argparse
import pytz
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt


def get_kolkata_time() -> datetime:
    """
    Get current time in Asia/Kolkata timezone.
    Returns:
        datetime object in Asia/Kolkata timezone
    """
    kolkata_tz = pytz.timezone('Asia/Kolkata')
    return datetime.now(kolkata_tz)


def is_within_operating_hours(start_hour: int = 9, end_hour: int = 19) -> bool:
    """
    Check if current time is within operating hours using Asia/Kolkata timezone.
    Args:
        start_hour: Start hour in 24-hour format (default: 9 for 9AM)
        end_hour: End hour in 24-hour format (default: 19 for 7PM)
    Returns:
        True if current time is within operating hours, False otherwise
    """
    kolkata_time = get_kolkata_time()
    current_hour = kolkata_time.hour
    return start_hour <= current_hour < end_hour


def load_config(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("config.json must be a list of camera objects")
        required = ["name", "zone", "rtsp"]
        for i, cam in enumerate(data):
            for k in required:
                if k not in cam:
                    raise ValueError(f"Camera index {i} missing key: {k}")
        return data


def open_rtsp_and_read_frame(rtsp: str, timeout_sec: float = 8.0, tries: int = 2):
    """
    Open RTSP and read a single frame within timeout. Returns frame (np.ndarray) or None.
    """
    if not rtsp:
        return None

    for attempt in range(tries):
        cap = None
        try:
            # Use CAP_FFMPEG with specific parameters for better RTSP handling
            cap = cv2.VideoCapture(rtsp)

            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set timeout for connection
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_sec * 1000))
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_sec * 1000))

            if not cap.isOpened():
                print(f"[RTSP] Failed to open stream: {rtsp} (attempt {attempt + 1}/{tries})")
                if cap:
                    cap.release()
                time.sleep(1)
                continue

            start = time.time()
            frame = None
            frame_attempts = 0
            max_frame_attempts = 5

            while time.time() - start < timeout_sec and frame_attempts < max_frame_attempts:
                ret, frm = cap.read()
                frame_attempts += 1

                if ret and frm is not None:
                    frame = frm
                    break

                time.sleep(0.1)

            if frame is not None:
                cap.release()
                return frame
            else:
                print(f"[RTSP] No frame received from: {rtsp} (attempt {attempt + 1}/{tries})")

        except Exception as e:
            print(f"[RTSP] Exception while reading from {rtsp}: {e} (attempt {attempt + 1}/{tries})")
        finally:
            if cap:
                cap.release()

        # Wait before retrying
        if attempt < tries - 1:
            time.sleep(1)

    print(f"[RTSP] Failed to get frame from {rtsp} after {tries} attempts")
    return None


def count_persons_in_frame(model: YOLO, frame, imgsz: int) -> int:
    """
    Run YOLOv8 on a single frame and return number of 'person' detections (COCO class 0).
    """
    results = model.predict(source=frame, verbose=False, imgsz=imgsz)
    if not results:
        return 0
    r = results[0]
    if r.boxes is None or r.boxes.cls is None:
        return 0
    cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int).tolist()
    return sum(1 for cid in cls_ids if cid == 0)


def build_payload_by_nvr(cam_cfgs: List[Dict[str, Any]], counts: List[int]) -> List[Dict[str, Any]]:
    """
    Build payload grouping cameras by NVR with the new format:
    {
        "DeviceID": "Lumina",
        "gatewaySerialNumber": "GV-3602-K",
        "gatewayName": "Lumina",
        "Date": "29092025",
        "Time": "131926",
        "points": [...]
    }
    """
    from collections import defaultdict

    # Group cameras by NVR
    nvr_groups = defaultdict(list)
    for i, cam in enumerate(cam_cfgs):
        nvr_name = cam.get("nvr", "Unknown")
        nvr_groups[nvr_name].append((cam, counts[i]))

    # Get current date and time in Asia/Kolkata timezone
    kolkata_time = get_kolkata_time()
    date_str = kolkata_time.strftime("%d%m%Y")  # DDMMYYYY format
    time_str = kolkata_time.strftime("%H%M%S")  # HHMMSS format

    payloads = []

    for nvr_name, cam_count_pairs in nvr_groups.items():
        # Get NVR serial from the first camera (assuming all cameras in same NVR have same serial)
        nvr_serial = cam_count_pairs[0][0].get("nvr_serial", "")

        points = []
        for cam, count in cam_count_pairs:
            point = {
                "name": cam.get("name"),
                "floor": cam.get("floor"),
                "zone": cam.get("zone"),
                "count": str(int(count if count is not None else 0))
            }
            points.append(point)

        payload = {
            "DeviceID": nvr_name,
            "gatewaySerialNumber": nvr_serial,
            "gatewayName": nvr_name,
            "Date": date_str,
            "Time": time_str,
            "points": points
        }
        payloads.append(payload)

    return payloads


def build_payload(cam_cfgs: List[Dict[str, Any]], counts: List[int]) -> List[Dict[str, Any]]:
    """
    Build payload preserving fields and adding ISO-8601 UTC timestamp.
    Keep 'count' as string per requested schema.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    payload: List[Dict[str, Any]] = []
    for cam, cnt in zip(cam_cfgs, counts):
        obj: Dict[str, Any] = {
            "nvr": cam.get("nvr"),
            "nvr_serial": cam.get("nvr_serial"),
            "name": cam.get("name"),
            "floor": cam.get("floor"),
            "zone": cam.get("zone"),
            "count": str(int(cnt if cnt is not None else 0)),
            "timestamp": now_iso,
        }
        payload.append(obj)
    return payload


def mqtt_publish(
    broker: str,
    port: int,
    topic: str,
    payload,
    username: Optional[str] = None,
    password: Optional[str] = None,
    tls: bool = False,
    keepalive: int = 60,
    qos: int = 0,
    retain: bool = False,
) -> None:
    """
    Publish with Paho v2. Adds a unique client_id.
    """
    msg = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)

    # generate unique client id (e.g. people-counter-<uuid4>)
    client_id = f"people-counter-{uuid.uuid4().hex[:8]}"

    client = mqtt.Client(
        client_id=client_id,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        protocol=mqtt.MQTTv311,
        transport="tcp",
    )

    if username:
        client.username_pw_set(username, password)
    if tls:
        client.tls_set()

    def on_connect(*args, **kwargs):
        try:
            client_, userdata, flags, rc = args[:4]
        except Exception:
            print("[MQTT] connect (args len:", len(args), ")")
            return
        print(f"[MQTT] connect rc={rc}, client_id={client_id}")

    def on_disconnect(*args, **kwargs):
        try:
            client_, userdata, rc = args[:3]
        except Exception:
            print("[MQTT] disconnect (args len:", len(args), ")")
            return
        print(f"[MQTT] disconnect rc={rc}, client_id={client_id}")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    client.connect(broker, port, keepalive)
    client.loop_start()
    try:
        info = client.publish(topic, payload=msg, qos=qos, retain=retain)
        info.wait_for_publish(timeout=3.0)
    finally:
        client.loop_stop()
        try:
            client.disconnect()
        except Exception as e:
            print(f"[MQTT] disconnect error: {e}")

    byte_len = len(msg.encode("utf-8")) if isinstance(msg, str) else 0
    print(f"[MQTT] published to '{topic}' (bytes={byte_len}) by {client_id}")


def run_once(
    cams: List[Dict[str, Any]],
    model: YOLO,
    imgsz: int,
    rtsp_timeout: float,
    rtsp_tries: int,
) -> List[Dict[str, Any]]:
    counts: List[int] = []
    for cam in cams:
        frame = open_rtsp_and_read_frame(
            cam.get("rtsp", ""),
            timeout_sec=rtsp_timeout,
            tries=rtsp_tries,
        )
        if frame is None:
            counts.append(0)
            continue
        cnt = count_persons_in_frame(model, frame, imgsz)
        counts.append(cnt)
    return build_payload_by_nvr(cams, counts)


def main():
    parser = argparse.ArgumentParser(
        description="Looped YOLOv8 single-frame person count per camera and publish to MQTT."
    )
    parser.add_argument("--config", default="config.json", help="Path to camera config JSON.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLOv8 model (e.g., yolov8n.pt/yolov8s.pt).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640).")
    parser.add_argument("--interval", type=int, default=20, help="Loop interval in seconds (default: 20).")
    parser.add_argument("--rtsp-timeout", type=float, default=8.0, help="Seconds to wait per RTSP for a frame.")
    parser.add_argument("--rtsp-tries", type=int, default=2, help="How many times to re-open RTSP if first try fails.")

    # Operating hours
    parser.add_argument("--start-hour", type=int, default=9, help="Start hour for inference (24-hour format, default: 9 for 9AM).")
    parser.add_argument("--end-hour", type=int, default=19, help="End hour for inference (24-hour format, default: 19 for 7PM).")

    # MQTT
    parser.add_argument("--mqtt-broker", default=os.getenv("MQTT_BROKER", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--mqtt-topic", default=os.getenv("MQTT_TOPIC", "cameras/person_counts"))
    parser.add_argument("--mqtt-user", default=os.getenv("MQTT_USER", ""))
    parser.add_argument("--mqtt-pass", default=os.getenv("MQTT_PASS", ""))
    parser.add_argument("--mqtt-tls", action="store_true", help="Enable TLS for MQTT connection.")
    parser.add_argument("--mqtt-qos", type=int, default=int(os.getenv("MQTT_QOS", "0")), choices=[0, 1, 2])
    parser.add_argument("--mqtt-retain", action="store_true", help="Publish retained message.")
    args = parser.parse_args()

    # Validate operating hours
    if args.start_hour < 0 or args.start_hour > 23:
        raise ValueError("Start hour must be between 0 and 23")
    if args.end_hour < 0 or args.end_hour > 23:
        raise ValueError("End hour must be between 0 and 23")
    if args.start_hour >= args.end_hour:
        raise ValueError("Start hour must be less than end hour")

    cams = load_config(args.config)
    model = YOLO(args.model)

    print(f"[INFO] Starting loop: interval={args.interval}s, cameras={len(cams)}")
    print(f"[INFO] Operating hours: {args.start_hour:02d}:00 - {args.end_hour:02d}:00 (Asia/Kolkata)")

    try:
        while True:
            # Check if current time is within operating hours
            if not is_within_operating_hours(args.start_hour, args.end_hour):
                kolkata_time = get_kolkata_time()
                current_time = kolkata_time.strftime("%H:%M:%S")
                print(f"[INFO] Outside operating hours ({current_time} IST), skipping inference. Next check in {args.interval}s")
                time.sleep(max(1, args.interval))
                continue

            payloads = run_once(
                cams=cams,
                model=model,
                imgsz=args.imgsz,
                rtsp_timeout=args.rtsp_timeout,
                rtsp_tries=args.rtsp_tries,
            )

            # Publish each NVR payload separately
            for payload in payloads:
                # Create topic with NVR name for better organization
                nvr_topic = f"{args.mqtt_topic}/{payload['DeviceID']}"

                mqtt_publish(
                    broker=args.mqtt_broker,
                    port=args.mqtt_port,
                    topic=nvr_topic,
                    payload=payload,
                    username=args.mqtt_user or None,
                    password=args.mqtt_pass or None,
                    tls=args.mqtt_tls,
                    qos=args.mqtt_qos,
                    retain=args.mqtt_retain
                )

                # Console preview of the JSON payload for each NVR
                print(f"[{payload['DeviceID']}] Published:")
                print(json.dumps(payload, indent=2))

            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")


if __name__ == "__main__":
    main()
