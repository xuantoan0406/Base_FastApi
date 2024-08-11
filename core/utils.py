from sentence_transformers.util import cos_sim


def cosine_distance(tensor1, tensor2):
    return cos_sim(tensor1, tensor2).item()
