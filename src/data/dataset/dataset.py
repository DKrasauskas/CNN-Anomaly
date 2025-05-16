import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os
import shutil


def clear_folder(folder_path):
    # Iterate through all files and subdirectories in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path) if os.path.isfile(file_path) else shutil.rmtree(file_path)


def reshape_set(arr, stride = 1, kernel_size = 1):
    r""""
    Reshape the input array into a sliding window view.
    Parameters
    ----------
    arr : numpy.ndarray
        Input array to be reshaped.
    stride : int, optional
        Stride for the sliding window. The default is 1.
    kernel_size : int, optional     
        Size of the sliding window. The default is 1.
    Returns
    -------
    numpy.ndarray
        Reshaped array with sliding window view.
    """
    kernel = sliding_window_view(arr, window_shape=kernel_size, axis = -1)
    return kernel[::stride, :]

def generate_dataset(input_paths, input_labels_path, indices,  path = "src/data/dataset/data/input_", label_path = "src/data/dataset/labels/labels.csv", list_path = "src/data/dataset/list_path.txt", stride = 1000, kernel_size = 1000, flush_folder = True):    
    r"""
    Generate target dataset from measured data

    Parameters
    ----------
    input_paths : list of str
        List of paths to the input files.
    input_labels_path : str
        Path to the input labels file.
    path : str, optional
        Path to the output files. The default is "src/data/dataset/data/input_".
    label_path : str, optional
        Path to the output labels file. The default is "src/data/dataset/labels/labels.csv".
    list_path : str, optional
        Path to the list of input files. The default is "src/data/dataset/list_path.txt".
    stride : int, optional
        Stride for the sliding window. The default is 1000.
    kernel_size : int, optional
        Size of the sliding window. The default is 1000.
    flush_folder : bool, optional
        If True, clears the output folder before generating the dataset. The default is True.
    Returns
    -------
    """

    if flush_folder:
        clear_folder("src/data/dataset/data")
        clear_folder("src/data/dataset/labels")
    labels = np.loadtxt(input_labels_path, dtype = np.float64, delimiter = ",")
    src_paths = open(list_path, 'w')
    write_labels = open(label_path, 'w')
    print(input_paths)
    for main_index, i in enumerate(input_paths):
        file = np.array(open(i, 'r', encoding='utf-8-sig').read().split("\n"))
        file = file[1:-1]
        label_current = labels[indices[main_index]-1][1]
        arr = reshape_set(file, stride = stride, kernel_size = kernel_size)
        for index, item in enumerate(arr):
            fs = "\n".join(item)
            current_index = index + len(arr) * main_index
            current_path = path + str(current_index)  + ".csv"
            write = open(current_path, 'w')
            src_paths.write(current_path + "\n")
            write.write(fs)
            write.close()
            write_labels.write(str(current_index) +"," + str(label_current) + "\n")
    src_paths.close()
    write_labels.close()

