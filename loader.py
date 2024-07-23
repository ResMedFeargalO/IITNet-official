import os
import torch
import numpy as np
from torch.utils.data import Dataset

from scipy.signal import butter, sosfilt

def fast_resample(xf,fs_out,fs_in):
	#start_time = time.time()
	if fs_in!=fs_out:    	
        
		ti = np.arange(0,(len(xf))/fs_in,1/fs_in)
		to = np.arange(0,(len(xf))/fs_in,1/fs_out)
        
		if(len(ti) != len(xf)):
			ti = ti[0:len(xf)]
            
		y = np.interp(to, ti, xf)  

	else:
		y = xf
	#print("---%s seconds for resample---"%(time.time()-start_time))
	return y

def filter_down(eeg_data, lowcut=.05, highcut=30, order=10, sample_in=100, sample_out=60):

    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * sample_in

    # Calculate the filter coefficients
    sos = butter(order, [lowcut / nyquist_freq, highcut / nyquist_freq], btype='bandpass', output='sos')

    # Apply the filter to the EEG data
    filtered_eeg = sosfilt(sos, eeg_data)

    fandd=fast_resample(filtered_eeg, fs_out=sample_out, fs_in=sample_in)
    return fandd
class EEGDataLoader(Dataset):

    def __init__(self, config, fold, mode='train'):

        self.mode = mode
        self.fold = fold

        self.config = config
        self.dataset = config['dataset']
        self.seq_len = config['seq_len']
        self.n_splits = config['n_splits']
        self.target_idx = config['target_idx']
        self.signal_type = config['signal_type']
        self.sampling_rate = config['sampling_rate']

        if self.dataset != "None": self.dataset_path = os.path.join('./datasets', self.dataset)
        else: self.dataset_path = '\\\\ir1-nasrst-p03\\nasuni\\Bru\\static\\lab_data\\Nyx\\Overnight' #OvernightsFOS_proc'
             
        self.inputs, self.labels, self.epochs = self.split_dataset()
        
    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        self.sampling_rate=60
        n_sample = 30 * self.sampling_rate * self.seq_len
        file_idx, idx, seq_len = self.epochs[idx]
        inputs = self.inputs[file_idx][idx:idx+seq_len]#[::2]

        inputs=filter_down(inputs.reshape(-1,))
        inputs = inputs.reshape(1, n_sample)
        #inputs=inputs[:,::2]

        inputs = torch.from_numpy(inputs).float()
        
        labels = self.labels[file_idx][idx:idx+seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]
        
        return inputs, labels

    def split_dataset(self):

        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.signal_type)
        data_fname_list = sorted(os.listdir(data_root))
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dataset)), allow_pickle=True)
        
        assert len(split_idx_list) == self.n_splits
        
        if self.dataset == 'Sleep-EDF':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx == self.fold - 1:
                    data_fname_dict['test'].append(data_fname_list[i])
                elif subject_idx in split_idx_list[self.fold - 1]:
                    data_fname_dict['val'].append(data_fname_list[i])
                else:
                    data_fname_dict['train'].append(data_fname_list[i])
                    
        elif self.dataset == 'MASS' or self.dataset == 'SHHS' or '{}':
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.mode]:
                    data_fname_dict[self.mode].append(data_fname_list[i])
            
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))
        
        for data_fname in data_fname_dict[self.mode]:
            npz_file = np.load(os.path.join(data_root, data_fname))
            inputs.append(npz_file['x'])
            labels.append(npz_file['y'])
            for i in range(len(npz_file['y']) - self.seq_len + 1):
                epochs.append([file_idx, i, self.seq_len])
            file_idx += 1
        
        return inputs, labels, epochs
    
class EEGDataLoader2(Dataset):
    def __init__(self, config, fold, mode='train'):

        self.mode = mode
        self.fold = fold

        self.config = config
        self.dataset = config['dataset']
        self.seq_len = config['seq_len']
        self.n_splits = config['n_splits']
        self.target_idx = config['target_idx']
        self.signal_type = config['signal_type']
        self.sampling_rate = config['sampling_rate']

        if self.dataset != "None": self.dataset_path = os.path.join('./datasets', self.dataset)
        else: self.dataset_path = '\\\\ir1-nasrst-p03\\nasuni\\Bru\\static\\lab_data\\Nyx\\OvernightsFOS_proc'
            
        self.inputs, self.labels, self.epochs = self.split_dataset()
        
    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        try:
            n_sample = 30 * self.sampling_rate * self.seq_len
            file_idx, idx, seq_len = self.epochs[idx]
            inputs = self.inputs[0][idx:idx+seq_len]
            #inputs=filter_down(inputs.reshape(-1,))
            #print(idx)
            #print(len(inputs[0]))
            print(np.max(np.array(self.epochs)))


            inputs = inputs.reshape(1, n_sample)
            inputs = torch.from_numpy(inputs).float()
            
            # labels = self.labels[file_idx][idx:idx+seq_len]
            # labels = torch.from_numpy(labels).long()
            # labels = labels[self.target_idx]
        except:
            return torch.from_numpy(np.zeros((1, n_sample))).float(), 0
        
        return inputs, 0#labels

    def split_dataset(self):
        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.signal_type)
        data_fname_list = sorted(os.listdir(data_root))
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dataset)), allow_pickle=True)
        
        for data_fname in data_fname_list:
            npz_file = np.load(os.path.join(data_root, data_fname))
            inputs.append(npz_file['x'])
            #labels.append(npz_file['y'])
            for i in range(len(npz_file['x']) - self.seq_len + 1):
                epochs.append([file_idx, i, self.seq_len])
            file_idx += 1
        
        return inputs, labels, epochs
