import matplotlib.pyplot as plt
import numpy as np
import re
import torch.utils.data as data
import torch
import src.settings.config as cfg
from operator import attrgetter


class Element:
    def __init__(self, x, y):
        raise NotImplementedError("Subclasses should implement this!")
    def preprocess():
        raise NotImplementedError("Subclasses should implement this!")

class Dataset:
    def __init__(self):
      raise NotImplementedError("Subclasses should implement this!")
    def load_data(self):
       raise NotImplementedError("Subclasses should implement this!")
       
class Voltage(Element):
    def __init__(self, data, preprocess = True):
        self.data = data
        self.__preprocess() if preprocess  else self.__preprocesLabels()
        self.__split_set()

    def __preprocess(self):
        self.data[..., 1] -= np.mean(self.data[..., 1])
    def __preprocesLabels(self):       
        self.data[..., 1] /= 1000
       
    def __split_set(self):
        self.x, self.y = self.data[:, 0], self.data[:, 1]


class VoltageDataset(Dataset):
    def __init__(self, paths, labels_path = None, dtype = torch.float32):
        """
        Dataset for the voltage data.
        Parameters
        ----------
        paths : list of str
            List of paths to the input files.
        labels_path : str, optional
            Path to the input labels file. The default is None.
        dtype : torch.dtype, optional
            Data type for the input data. The default is torch.float32.
        Returns
        -------
        """
        self.paths = paths
        self.labels_path = labels_path
        self.set = np.empty(len(paths), dtype=object)
        self.size = 0
        self.dtype = dtype
        self.__load_data()
        self.__to_tensor()
    
    def __load_data(self):
        """
        This is a private method.
        Load the data from the input files and store it in the set attribute.
        The data is preprocessed and split into x and y attributes.
        The labels are loaded from the labels_path attribute.
        The size of the dataset is calculated based on the number of elements in the set.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        for i, p in enumerate(self.paths):
            self.set[i] = Voltage(np.loadtxt(p, delimiter=','))       
            self.size += len(self.set[i].data) * 2 * 4 #float32
        self.labels = Voltage(np.loadtxt(self.labels_path, delimiter=','), preprocess=False)
        
    def __to_tensor(self):
        """
        This is a private method.
        Convert the data to tensor format and store it in the input_x, input_y, and target attributes.
        The input_x and input_y attributes are the x and y attributes of the set attribute.
        The target attribute is the labels attribute.
        The data is moved to the device specified in the cfg.DEVICE attribute.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if cfg.MEM_MAX < self.size:
            raise Exception("Memory limit exceeded")
        x = np.array(list(map(attrgetter('x'), self.set)))
        y = np.array(list(map(attrgetter('y'), self.set)))
        labels = np.array(self.labels.y)
        self.input_x = torch.tensor(x, dtype = self.dtype).to(cfg.DEVICE)
        self.input_y = torch.tensor(y, dtype=  self.dtype).to(cfg.DEVICE) 
        self.target = torch.tensor(labels[:len(self.paths)], dtype = self.dtype).to(cfg.DEVICE)
        self.dataset = data.TensorDataset(self.input_y, self.target)

    def info(self, cuda_detailed = False, log = None):
        """
        Print the information about the dataset.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if log is None:
            if(cuda_detailed):
                print("CUDA details:")
                print(f"CUDA device: {cfg.DEVICE}")
                print(f"CUDA device name: {torch.cuda.get_device_name(cfg.DEVICE)}")
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated(cfg.DEVICE) * 0.000001:.2f} MB")
                print(f"CUDA memory cached: {torch.cuda.memory_reserved(cfg.DEVICE) * 0.000001:.2f} MB")
                print(f"CUDA memory max: {cfg.MEM_MAX * 0.000001:.2f} MB")
            else:
                print(f"Memory allocated -> {torch.cuda.memory_allocated(cfg.DEVICE) * 0.000001:.2f} MB")
        else:      
            log.log_general("CUDA details:")
            log.log_general(f"CUDA device: {cfg.DEVICE}")
            log.log_general(f"CUDA device name: {torch.cuda.get_device_name(cfg.DEVICE)}")
            log.log_general(f"CUDA memory allocated: {torch.cuda.memory_allocated(cfg.DEVICE) * 0.000001:.2f} MB")
            log.log_general(f"CUDA memory cached: {torch.cuda.memory_reserved(cfg.DEVICE) * 0.000001:.2f} MB")
            log.log_general(f"CUDA memory max: {cfg.MEM_MAX * 0.000001:.2f} MB")
            log.log_general(f"Memory allocated -> {torch.cuda.memory_allocated(cfg.DEVICE) * 0.000001:.2f} MB")
            

       
 

   