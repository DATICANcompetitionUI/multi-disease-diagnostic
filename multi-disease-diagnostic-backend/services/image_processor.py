import io

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = (64, 64)


def preprocess_image(image_bytes: bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = ImageOps.exif_transpose(image)

    image = ImageOps.fit(
        image,
        IMG_SIZE,
        method=Image.Resampling.LANCZOS,
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    return np.expand_dims(image, axis=0)