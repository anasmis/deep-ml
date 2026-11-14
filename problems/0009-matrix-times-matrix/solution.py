def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    import numpy as np
    arr = np.array(a)
    brr = np.array(b)
    try:
        c = arr@brr
    except Exception as e:
        return -1
    return c