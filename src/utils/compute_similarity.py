import numpy as np
from config import get_similarity_measure

import numpy as np


def compute_cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray):

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length")

    dot_product = np.dot(vector_a, vector_b)

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


def compute_similarity(x: np.ndarray, y: np.ndarray) -> float:
    similarity_measure = get_similarity_measure()

    if similarity_measure == "cosine":
        return compute_cosine_similarity(x, y)
    elif similarity_measure == "dot":
        return np.dot(x, y)

    raise ValueError("Invalid similarity measure")
