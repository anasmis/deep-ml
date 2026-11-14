def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # Return a list where each element is the dot product of a row of 'a' with 'b'.
    # If the number of columns in 'a' does not match the length of 'b', return -1.
    import numpy as np
    arr = np.array(a)
    brr = np.array(b)
    if len(a[0]) == len(b):
        return [a_i.dot(brr) for a_i in arr]
    return -1