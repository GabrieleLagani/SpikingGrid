import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from bindsnet.datasets import DataLoader
from bindsnet.encoding import NullEncoder

import params as P


class IntensityTransform:
    def __init__(self, intensity, inpt_norm):
        self.intensity = intensity
        self.inpt_norm=inpt_norm

    def __call__(self, x):
        return x * (self.inpt_norm / x.norm(p=2) if self.inpt_norm is not None else 1.) * self.intensity

class MNISTWrapper(MNIST):
    def __init__(self, image_encoder=None, subset_idx=None, classes=None, label_shape=None, label_intensity=None, assignments=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        
        # Image encoder into spikes
        self.image_encoder = image_encoder if image_encoder is not None else NullEncoder()

        # Only use subset of desired size
        if subset_idx is not None:
            self.targets = self.targets[subset_idx]
            self.data = self.data[subset_idx]

        # Only use subset of classes
        if classes is not None:
            self.classes = classes
            # keep only samples whose class is in the desired set of classes
            idx = (self.targets == self.classes[0])
            for i in range(1, len(self.classes)): idx |= (self.targets == self.classes[i])
            self.targets = self.targets[idx]
            self.data = self.data[idx]

        # Determine regions of the input grid to be associated with label signal
        self.label_shape = None
        self.label_intensity = None
        self.assignments = None
        if assignments is not None and label_intensity is not None and label_shape is not None:
            self.label_shape = label_shape
            self.label_intensity = label_intensity
            self.assignments = {c: (assignments == c).float() * self.label_intensity for c in self.classes}

    def __getitem__(self, ind):
        image, label = super().__getitem__(ind)

        # Extend image with label information
        if self.label_shape is not None:
            image = torch.cat((image, torch.zeros(self.label_shape)), 2) + self.assignments[label]
        
        output = {
            "encoded_image": self.image_encoder(image),
            "label": label,
        }

        return output

    def __len__(self):
        return super().__len__()

class DataManager:
    def __init__(self, n_train, n_eval, inpt_shape, grid_shape, label_shape, assignments, inpt_norm, intensity, label_intensity):
        self.n_train = min(n_train, P.TRN_SET_SIZE)
        self.n_eval = min(n_eval, P.TST_SET_SIZE)
        self.validate_on_tst_set = P.TRN_SET_SIZE - self.n_train < self.n_eval
        self.inpt_shape = inpt_shape
        self.grid_shape = grid_shape
        self.label_shape = label_shape
        self.assignments = assignments
        self.inpt_norm = inpt_norm
        self.intensity = intensity
        self.label_intensity = label_intensity
        
        # Define data transformations
        self.T = transforms.Compose([transforms.Resize(self.inpt_shape[1]), transforms.ToTensor(), IntensityTransform(self.intensity, self.inpt_norm)])

    def get_train(self, classes, encoder, batch_size):
        dataset = MNISTWrapper(
            encoder, subset_idx=range(self.n_train), classes=classes, label_shape=self.label_shape, label_intensity=self.label_intensity,
            assignments=self.assignments, root=P.DATA_FOLDER, train=True, download=True, transform=self.T
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=P.N_WORKERS, pin_memory=P.GPU)
    
    def get_train4eval(self, classes, encoder, batch_size):
        dataset = MNISTWrapper(
            encoder, subset_idx=range(min(self.n_train, self.n_eval)), classes=classes, label_shape=self.label_shape, label_intensity=0.,
            assignments=self.assignments, root=P.DATA_FOLDER, train=True, download=True,  transform=self.T
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=P.N_WORKERS, pin_memory=P.GPU)
    
    def get_val(self, classes, encoder, batch_size):
        if self.validate_on_tst_set: return self.get_test(classes, encoder, batch_size)
        dataset = MNISTWrapper(
            encoder, subset_idx=range(P.TRN_SET_SIZE-self.n_eval, P.TRN_SET_SIZE), classes=classes, label_shape=self.label_shape, label_intensity=0.,
            assignments=self.assignments, root=P.DATA_FOLDER, train=True, download=True,  transform=self.T
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=P.N_WORKERS, pin_memory=P.GPU)
    
    def get_test(self, classes, encoder, batch_size):
        dataset = MNISTWrapper(
            encoder, subset_idx=range(self.n_eval), classes=classes, label_shape=self.label_shape, label_intensity=0.,
            assignments=self.assignments, root=P.DATA_FOLDER, train=False, download=True, transform=self.T
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=P.N_WORKERS, pin_memory=P.GPU)