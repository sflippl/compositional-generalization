import numpy as np
import pandas as pd
import functools

def array_to_dataframe(array):
    dims = array.shape
    flat_array = array.flatten()
    dct_flat = {
        "dim%d"%i: np.array(
            np.repeat(
                range(dims[i]), functools.reduce(lambda x,y:x*y, dims[i+1:], 1)
            ).tolist() * functools.reduce(lambda x,y:x*y, dims[:i], 1)
        ) for i in range(len(dims))
    }
    dct_flat['array'] = flat_array
    df_flat = pd.DataFrame(dct_flat)
    return df_flat