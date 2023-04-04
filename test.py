import pickle
import numpy as np

if __name__ == '__main__':
    # load the interpolation function from a file
    with open('ac_function.pkl', 'rb') as f_in:
        f = pickle.load(f_in)
    p=np.array([3,0.3])
    print(f(p))