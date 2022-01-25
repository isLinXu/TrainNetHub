from typing import Union, Tuple, List

import cv2
import numpy as np


def draw_rectangle(img: np.ndarray, box: Union[Tuple, List], color: Tuple = (255, 0, 0), thickness: int = 3):
    img = cv2.rectangle(
        img,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        color=color, thickness=thickness,
    )

    return img

def put_text(
        img: np.ndarray,
        text: str,
        coords: Tuple[int, int],
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.7,
        color=(255, 255, 255),
        thickness: int = 2,
        line_type=cv2.LINE_AA,
) -> None:
    cv2.putText(
        img, text,
        (int(coords[0]), int(coords[1])),
        font_face, font_scale,
        color, thickness,
        line_type,
    )
