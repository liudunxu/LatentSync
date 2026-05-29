from insightface.app import FaceAnalysis
import numpy as np
import torch

INSIGHTFACE_DETECT_SIZE = 512


class FaceDetector:
    def __init__(self, device="cuda"):
        self.app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            root="checkpoints/auxiliary",
            providers=["CUDAExecutionProvider"],
        )
        self.app.prepare(ctx_id=cuda_to_int(device), det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))

    def __call__(self, frame, threshold=0.5):
        f_h, f_w, _ = frame.shape

        faces = self.app.get(frame)

        if len(faces) == 0:
            return None, None

        valid_faces = []
        for face in faces:
            bbox = face.bbox.astype(np.int_).tolist()
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w < 30 or h < 40:
                continue
            if w / h > 2.0 or w / h < 0.15:
                continue
            if face.det_score < threshold:
                continue
            valid_faces.append((face, w * h))

        if not valid_faces:
            return None, None

        valid_faces.sort(key=lambda x: x[1], reverse=True)
        face, _ = valid_faces[0]
        lmk = np.round(face.landmark_2d_106).astype(np.int_)

        halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)

        sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
        halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
        upper_bond = halk_face_coord[1] - halk_face_dist

        x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond), np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()

        y2 += int((x2 - x1) * 0.1)
        x1 -= int((x2 - x1) * 0.05)
        x2 += int((x2 - x1) * 0.05)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(f_w, x2)
        y2 = min(f_h, y2)
        return (x1, y1, x2, y2), lmk


def cuda_to_int(cuda_str: str) -> int:
    """
    Convert the string with format "cuda:X" to integer X.
    """
    if cuda_str == "cuda":
        return 0
    device = torch.device(cuda_str)
    if device.type != "cuda":
        raise ValueError(f"Device type must be 'cuda', got: {device.type}")
    return device.index


LMK_ADAPT_ORIGIN_ORDER = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    73,
    74,
    86,
]
