import torch 
import numpy as np
import src.settings.config as cfg
class SimpleCNN(torch.nn.Module):
    """
    This is a network model proven to work on the 50s long raw voltage dataset with a 80% accuracy. Hyperparameters tuned manualy.
    """
    def __init__(self, input_size,  output_size):
        """
        Parameters
        ----------
        input_size : Deprecated
            as this model is trained exclusively on the 50s long raw voltage dataset, this parameter is not used.
        output_size : int
            Number of output classes. One is used for the current case.
        Returns
        -------
        None.
        """
        super(SimpleCNN, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.conv1  = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=100, stride=10, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)  # Reduce to (batch, channels, 1)
        self.fc1 = torch.nn.Linear(1999936   , 32)#4 * 1250001 8 * 1249502 
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, output_size)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ComplexCNN(torch.nn.Module):
    """
    This is a network model proven to work on the 50s long raw voltage dataset with a 95% accuracy. Hyperparameters tuned manualy.
    """
    def __init__(self, input_size,  output_size):
        """
        Parameters
        ----------
        input_size : Deprecated
            as this model is trained exclusively on the 50s long raw voltage dataset, this parameter is not used.
        output_size : int
            Number of output classes. One is used for the current case.
        Returns
        -------
        None.
        """
        super(ComplexCNN, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.conv0  = torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5000, stride=10, padding=1)
        self.conv1  = torch.nn.Conv1d(in_channels=8, out_channels=32, kernel_size=1000, stride=20, padding=1)
        self.conv2  = torch.nn.Conv1d(in_channels=32, out_channels=48, kernel_size=50, stride=2, padding=1)
        self.conv3  = torch.nn.Conv1d(in_channels=48, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.dropout = torch.nn.Dropout(p=0.0)
        self.fc1 = torch.nn.Linear(12160, 32)#4 * 1250001 8 * 1249502 
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, output_size)

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = x.view(-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class WindowedCNN(torch.nn.Module):
    """
    This is the main model
    """
    def __init__(self, input_size,  output_size):
        """
        Parameters
        ----------
        input_size : input signal size
        output_size : int
            Number of output classes. One is used for the current case.
        Returns
        -------
        None.
        """
        super(WindowedCNN, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.conv0  = torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=500, stride=10, padding=1)
        self.conv1  = torch.nn.Conv1d(in_channels=8, out_channels=32, kernel_size=100, stride=20, padding=1)
        self.conv2  = torch.nn.Conv1d(in_channels=32, out_channels=48, kernel_size=50, stride=2, padding=1)
        self.conv3  = torch.nn.Conv1d(in_channels=48, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.dropout = torch.nn.Dropout(p=0.0)
        self.conv0_outsize = np.floor((input_size - 500 + 1 + 2 * 1) / 10)
        self.conv1_outsize = np.floor((np.ceil(self.conv0_outsize / 2) - 100 + 1 + 2 * 1) / 20)
        self.conv2_outsize = np.floor((np.ceil(self.conv1_outsize / 2) - 50 + 1 + 2 * 1) / 2)
        self.conv3_outsize = np.floor((np.ceil(self.conv2_outsize / 2) - 5 + 1 + 2 * 1) / 2)
        self.fcc_insize = int(np.ceil(self.conv3_outsize /2)* 64)
        self.fc1 = torch.nn.Linear(320, 32)#4 * 1250001 8 * 1249502 
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, output_size)


    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = x.view(-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def info(self, detailed = False, cuda_detailed = False):
        """
        Print the information about the Network.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if(detailed):
            print("Network details:")
            print(f"Conv0 input size: {self.conv0_outsize}")
            print(f"Conv1 input size: {self.conv1_outsize}")
            print(f"Conv2 input size: {self.conv2_outsize}")
            print(f"Conv3 input size: {self.conv3_outsize}")
            print(f"FCC input size: {self.fcc_insize}")
        if(cuda_detailed):
            print("CUDA details:")
            print(f"CUDA device: {cfg.DEVICE}")
            print(f"CUDA device name: {torch.cuda.get_device_name(cfg.DEVICE)}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated(cfg.DEVICE) * 0.000001:.2f} MB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved(cfg.DEVICE) * 0.000001:.2f} MB")
            print(f"CUDA memory max: {cfg.MEM_MAX * 0.000001:.2f} MB")
        else:
            print(f"Memory allocated -> {torch.cuda.memory_allocated(cfg.DEVICE) * 0.000001:.2f} MB")
       