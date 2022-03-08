def rgb_matrix2id_matrix(rgb_matrix, rgb2id: dict):
    return [
        [rgb2id[each] for each in line] for line in rgb_matrix
    ]