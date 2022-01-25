from pathlib import Path
from typing import Union, Tuple, Dict

import cv2
import numpy as np


def imread(path: Union[Path, str]) -> np.ndarray:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imwrite(path: Union[Path, str], img: np.ndarray) -> None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)


def get_video_capture(path: Union[Path, str]) -> Union[cv2.VideoCapture, Dict]:
    cap = cv2.VideoCapture(str(path))
    meta = dict(
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    return cap, meta


def get_video_writer(path: Union[Path, str], meta: Dict, **meta_rewrite_kwargs) -> cv2.VideoWriter:
    meta = meta.copy()
    for k, v in meta_rewrite_kwargs.items():
        meta[k] = v

    shape = (meta['width'], meta['height'])
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MP4V'), meta['fps'], shape)
    return out


def video_read_frame(cap: cv2.VideoCapture) -> Tuple[bool, np.ndarray]:
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return ret, frame


def video_retrieve_frame(cap: cv2.VideoCapture) -> Tuple[bool, np.ndarray]:
    ret, frame = cap.retrieve()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return ret, frame


def video_write_frame(out: cv2.VideoWriter, frame: np.ndarray) -> None:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out.write(frame)
