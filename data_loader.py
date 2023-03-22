from glob import glob
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from numpy import zeros
import numpy as np


class Train_WrittenDatset(Dataset):
    def __init__(self,root_dir):
        self.data = self.txt_tensor(root_dir)
        self.label = self.read_label(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def txt_tensor(self,path):
        file_list = sorted(glob.glob(path + "/train/*.txt"))
        alldata = []
        for file_path in file_list:
            with open(file_path, "r") as f:
                returnVect = zeros((1, 1024))
                for i in range(32):
                    lineStr = f.readline()
                    for j in range(32):
                        returnVect[0, 32 * i + j] = int(lineStr[j])
                alldata.append(returnVect)
        returntensor = torch.squeeze(torch.tensor(np.array(alldata)))
        returntensor = returntensor.to(torch.float32)
        # print(returntensor)
        return returntensor


    def read_label(self,root_dir):
        labels = np.loadtxt(open(root_dir+"/train.csv","rb"),delimiter=",",skiprows=1,usecols=[1])
        labels = labels.astype(int)
        return self.label2hot(labels,10)

    def label2hot(self,labels,num_classes):
        one_hot = torch.zeros(len(labels), num_classes)
        for i, label in enumerate(labels):
            one_hot[i][label] = 1
        # print(one_hot[:10])
        return one_hot


class Test_WrittenDatset(Dataset):
    def __init__(self,test_path):
        self.test_data = self.test_tensor(test_path)

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        return self.test_data[index]

    def test_tensor(self,test_path):
        file_list = sorted(glob.glob(test_path + "/*.txt"))
        alldata = []
        for file_path in file_list:
            with open(file_path, "r") as f:
                returnVect = zeros((1, 1024))
                for i in range(32):
                    lineStr = f.readline()
                    for j in range(32):
                        returnVect[0, 32 * i + j] = int(lineStr[j])
                alldata.append(returnVect)
        return_test_tensor = torch.squeeze(torch.tensor(np.array(alldata)))
        return_test_tensor = return_test_tensor.to(torch.float32)
        return return_test_tensor
