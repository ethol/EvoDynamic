import numpy as np


def int_to_binary_string(int_number, size):
    binary_array = list(str(int(bin(int_number)[2:])))
    binary_array = list(map(int, binary_array))
    return left_pad_array(binary_array, size)


def left_pad_array(arr, size):
    array = np.zeros(size, dtype=int)
    array[-len(arr):] = arr
    return array
