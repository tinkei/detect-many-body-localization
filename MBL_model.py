"""Model for Many Body Localization"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

import os
import gzip
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from MBL_dataset import MBLDataset


class MBLModel(pl.LightningModule):
    """Model for Many Body Localization"""

    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        # super(MBLModel09, self).__init__()
        super().__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################

        ########################################################################
        #                                Model                                 #
        ########################################################################

        self.cnnlayers = []
        self.layers = []

        print("Number of CNN layers:", len(self.hparams["layers_cnn"]))
        print("Number of FC layers:", len(self.hparams["layers_fc"]))

        if "print_layer_size" not in self.hparams:
            self.hparams["print_layer_size"] = False
        if "split_output" not in self.hparams:
            self.hparams["split_output"] = False
        if "split_correct" not in self.hparams:
            self.hparams["split_correct"] = True

        verbose = self.hparams["print_layer_size"]

        for idx, layer_hparams in enumerate(self.hparams["layers_cnn"]):

            if "use_avg_pool" not in layer_hparams:
                layer_hparams["use_avg_pool"] = False
            if "use_max_pool" not in layer_hparams:
                layer_hparams["use_max_pool"] = False
            if "dropout" not in layer_hparams:
                layer_hparams["dropout"] = 0
            if "padding" not in layer_hparams:
                layer_hparams["padding"] = 0
            if "groups" not in layer_hparams:
                layer_hparams["groups"] = 1
            # print(layer_hparams["in_channels"], layer_hparams["out_channels"], layer_hparams["groups"])

            layer_list = [
                nn.Conv2d(layer_hparams["in_channels"], layer_hparams["out_channels"], layer_hparams["kernel_size"], layer_hparams["stride"], layer_hparams["padding"], groups=layer_hparams["groups"], bias=False),
                nn.BatchNorm2d(layer_hparams["out_channels"]),
                nn.PReLU(layer_hparams["out_channels"]),
            ]
            if layer_hparams["dropout"] != 0:
                layer_list.append(
                    nn.Dropout2d(layer_hparams["dropout"])
                )
            if layer_hparams["use_avg_pool"]:
                layer_list.append(
                    nn.AvgPool2d(kernel_size=2)
                )
            if layer_hparams["use_max_pool"]:
                layer_list.append(
                    nn.MaxPool2d(kernel_size=2)
                )

            if "upsample" in layer_hparams and layer_hparams["upsample"]:
                temp_layer = nn.Upsample(scale_factor=2, mode="bicubic")
            else:
                temp_layer = nn.Sequential(*layer_list)

            # temp_layer.apply(self.init_weights)
            self.layers.append(temp_layer)


        temp_layer = nn.Flatten()
        self.layers.append(temp_layer)


        for idx, layer_hparams in enumerate(self.hparams["layers_fc"]):

            if "dropout" not in layer_hparams:
                layer_hparams["dropout"] = 0
            if "split_output" not in self.hparams:
                self.hparams["split_output"] = False

            if idx != len(self.hparams["layers_fc"]) - 1:
                if self.hparams["split_output"]:
                    keypoint_count = 15

                    num_channels = layer_hparams["in_features"]
                    channels_list = [i for i in range(num_channels)]
                    split_list = np.array_split(channels_list, keypoint_count)

                    num_channels_o = layer_hparams["out_features"]
                    channels_list_o = [i for i in range(num_channels_o)]
                    split_list_o = np.array_split(channels_list_o, keypoint_count)

                    for i in range(keypoint_count):
                        own_channels = split_list[i].tolist()
                        own_channels_o = split_list_o[i].tolist()
                        if verbose and False:
                            print("Linear layer:", i)
                            print("Orignal input/output size", [layer_hparams["in_features"], layer_hparams["out_features"]])
                            print("Split input/output size", [len(own_channels), len(own_channels_o)])
                        # Correct slicing:
                        if self.hparams["split_correct"]:
                            temp_layer = nn.Linear(len(own_channels), len(own_channels_o))
                        # Incorrect slicing:
                        else:
                            temp_layer = nn.Linear(layer_hparams["in_features"], len(own_channels_o))
                        self.layers.append(temp_layer)
                else:
                    temp_layer = nn.Sequential(
                        nn.Linear(layer_hparams["in_features"], layer_hparams["out_features"], bias=False),
                        nn.BatchNorm1d(layer_hparams["out_features"]),
                        nn.PReLU(layer_hparams["out_features"]),
                        nn.Dropout(layer_hparams["dropout"]),
                    )
                    self.layers.append(temp_layer)

            else:
                if self.hparams["split_output"]:
                    keypoint_count = 15

                    num_channels = layer_hparams["in_features"]
                    channels_list = [i for i in range(num_channels)]
                    split_list = np.array_split(channels_list, keypoint_count)

                    for i in range(keypoint_count):
                        own_channels = split_list[i].tolist()
                        # own_channels = range(layer_hparams["in_features"])
                        if verbose and False:
                            print("Last linear layer:", i)
                            print("Orignal input/output size", [layer_hparams["in_features"], layer_hparams["out_features"]])
                            print("Split input/output size", [len(own_channels), int(layer_hparams["out_features"] / keypoint_count)])
                        # Correct slicing:
                        if self.hparams["split_correct"]:
                            temp_layer = nn.Linear(len(own_channels), int(layer_hparams["out_features"] / keypoint_count))
                        # Incorrect slicing:
                        else:
                            temp_layer = nn.Linear(layer_hparams["in_features"], int(layer_hparams["out_features"] / keypoint_count))
                        self.layers.append(temp_layer)
                else:
                    temp_layer = nn.Linear(layer_hparams["in_features"], layer_hparams["out_features"])
                    self.layers.append(temp_layer)


        self.model = nn.ModuleList(self.layers)

        # for layer in self.layers:
        #     print(layer)

        # self.model = nn.Sequential(*self.layers)
        self.model.apply(self.init_weights)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def init_weights(self, l):
        if type(l) == nn.Linear or type(l) == nn.Conv2d:
            gain = nn.init.calculate_gain('leaky_relu', 0.25)
            nn.init.xavier_normal_(l.weight, gain=gain)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        # x = self.model(x)

        skip_x = None
        skip_counter = 0
        tensor_list = []
        split_output = []
        keypoint_count = 15
        for idx, layer in enumerate(self.model):

            if "print_layer_size" not in self.hparams:
                self.hparams["print_layer_size"] = False
            if "split_output" not in self.hparams:
                self.hparams["split_output"] = False
            if "split_correct" not in self.hparams:
                self.hparams["split_correct"] = True

            entry_count = self.hparams["entry_count"]
            group_count = self.hparams["group_count"]
            group_size = self.hparams["group_size"]
            pool_every = self.hparams["pool_every"]
            verbose = self.hparams["print_layer_size"]


            if idx < entry_count: # Initial CNN layers

                x = layer(x)
                if verbose:
                    print(idx, x.size())

            elif entry_count <= idx and idx < entry_count + group_count * group_size: # CNN blocks

                # y = x.detach().clone()

                # Heavily customized structure.
                pos = (idx - entry_count) % group_size

                if verbose and pos == 0:
                    print("===== Inception unit {} =====".format(int((idx - entry_count)//group_size + 1)))
                    # if skip_x is not None and skip_counter % pool_every == pool_every - 1:
                    if skip_x is not None and skip_counter % pool_every != 0:
                        print("Input", torch.cat([x, skip_x], dim=1).size())
                    else:
                        print("Input", x.size())
                        

                # First layer is 3x3 stride 1 with MaxPool
                if pos == 0:
                    # if skip_x is not None and skip_counter % pool_every == pool_every - 1:
                    if skip_x is not None and skip_counter % pool_every != 0:
                        y = layer(torch.cat([x, skip_x], dim=1))
                    else:
                        y = layer(x)
                    tensor_list.append(y)
                    if verbose:
                        print(idx, pos, y.size())
                # Second layer is two 3x3 stride 1 with MaxPool
                elif pos == 1:
                    # if skip_x is not None and skip_counter % pool_every == pool_every - 1:
                    if skip_x is not None and skip_counter % pool_every != 0:
                        y = layer(torch.cat([x, skip_x], dim=1))
                    else:
                        y = layer(x)
                elif pos == 2:
                    y = layer(y)
                    tensor_list.append(y)
                    if verbose:
                        print(idx, pos, y.size())

                # Last layer: Concat everything.
                if pos == group_size - 1:
                    x = torch.cat(tensor_list, dim=1)
                    if skip_counter % pool_every == 0:
                        if verbose:
                            print("Storing skip_x")
                        skip_x = x
                    if skip_counter % pool_every == pool_every - 1:
                        if verbose:
                            print("Removing skip_x")
                        skip_x = None
                    skip_counter += 1
                    tensor_list = []
                    if verbose:
                        print(idx, pos, x.size())
                        print("============================")
            
            elif self.hparams["split_output"] and idx >= len(self.model) - keypoint_count * len(self.hparams["layers_fc"]): # Final layer(s)

                pos = (keypoint_count - (len(self.model) - idx)) % keypoint_count

                if self.hparams["split_correct"]:
                    num_channels = x.size()[1]
                    channels_list = [i for i in range(num_channels)]
                    split_list = np.array_split(channels_list, keypoint_count)
                    own_channels = split_list[pos].tolist()
                    x_sliced = x.narrow(1,int(own_channels[0]),len(own_channels))
                    if verbose:
                        print("Final layers input", x.size(), num_channels)
                        # print(len(split_list))
                        # print( [(len(i), i[0], i[-1]) for i in split_list])
                        # print(x_sliced.size())
                    y = layer(x_sliced)
                else:
                    y = layer(x)
                split_output.append(y)
                if verbose:
                    print("Final layers", idx, pos, y.size())

                if pos == keypoint_count - 1:
                    x = torch.cat(split_output, dim=1)
                    split_output = []
                    if verbose:
                        print("Final layer", idx, x.size())

            else: # Flatten and FC layers

                x = layer(x)

        ########################################################################
        #                              END OF CODE                             #
        ########################################################################
        return x

    def general_step(self, batch, batch_idx, mode):

        inputs, targets = batch["image"], batch["label"]

        # Forward pass.
        outputs = self.forward(inputs)

        # Loss.
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(outputs, targets)

        # Predictions.
        preds = outputs.argmax(axis=1)
        n_correct = (preds == targets).sum()
        acc = 0 # n_correct / len(targets)
        return loss, n_correct, acc

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        # avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        # avg_score = torch.stack([x[mode + '_score'] for x in outputs]).mean().cpu().numpy()
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct, acc = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}#, 'train_acc': acc}
        progress_bar = {'n_correct': n_correct}
        return {'loss': loss, 'train_n_correct':n_correct, 'progress_bar': progress_bar, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, acc = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct':n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc' : acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    def load_rho_train(self, obj_name, L, n, periodic, num_EV, data_dir):
        """Load a list of reduced density matrices with W = {0.5, 8}.

        Parameters
        ----------
        obj_name : str
            A name you give to this object. Call it `rho_A`.
        L : int
            System size.
        n : int
            Number of consecutive spins sampled.
        periodic : bool
            Whether the Hamiltonian is periodic.
        num_EV : int
            Number of eigenvalues around zero being sampled.
        data_dir : str, optional
            Directory where the data is saved.

        Return
        ------
        obj : list
            A list of lists, where each reduced density matrix is paired with its disorder strength W.
            i.e. obj[i][0] is a 2D numpy.ndarray of the reduced density matrix, and obj[i][1] is the disorder strength used to generate it.
            Number of data must be a multiple of 10.
        """

        directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'n={:02d}'.format(n), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
        os.makedirs(directory, exist_ok=True)

        # Check if file exists, load the file, and increment suffix.
        i = 0
        data = []
        while os.path.exists(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i))):
            with gzip.open(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i)), 'rb') as handle:
                data = data + pickle.load(handle)
                if n >= 6 and len(data) >= 200000:
                    return data # Don't have enough RAM to load 400000 samples.
            i += 1

        return data

    def load_rho_valid(self, obj_name, L, n, periodic, num_EV, data_dir):
        """Load a list of reduced density matrices with random W != {0.5, 8}.

        Parameters
        ----------
        obj_name : str
            A name you give to this object. Call it `rho_A`.
        L : int
            System size.
        n : int
            Number of consecutive spins sampled.
        periodic : bool
            Whether the Hamiltonian is periodic.
        num_EV : int
            Number of eigenvalues around zero being sampled.
        data_dir : str, optional
            Directory where the data is saved.

        Return
        ------
        obj : list
            A list of lists, where each reduced density matrix is paired with its disorder strength W.
            i.e. obj[i][0] is a 2D numpy.ndarray of the reduced density matrix, and obj[i][1] is the disorder strength used to generate it.
            Number of data must be a multiple of 10.
        """

        directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'n={:02d}'.format(n), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
        os.makedirs(directory, exist_ok=True)

        # Check if file exists, load the file, and increment suffix.
        i = 0
        data = []
        while os.path.exists(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i))):
            with gzip.open(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i)), 'rb') as handle:
                data = data + pickle.load(handle)
                if n >= 6 and len(data) >= 200000:
                    return data # Don't have enough RAM to load 400000 samples.
            i += 1

        return data

    def prepare_data(self):

        obj_name = self.hparams['MBL']['obj_name']
        L        = self.hparams['MBL']['L']
        n        = self.hparams['MBL']['n']
        periodic = self.hparams['MBL']['periodic']
        num_EV   = self.hparams['MBL']['num_EV']
        rho_train_data_dir = self.hparams['MBL']['rho_train_data_dir']
        rho_valid_data_dir = self.hparams['MBL']['rho_valid_data_dir']

        data_train = self.load_rho_train(obj_name, L, n, periodic, num_EV, data_dir=rho_train_data_dir)
        data_valid = self.load_rho_valid(obj_name, L, n, periodic, num_EV, data_dir=rho_valid_data_dir)
        # if len(data_train) <= 10000:
        #     raise RuntimeError('Insufficient data. Training data has length {} <= 10000.'.format(len(data_train)))
        train_dataset = MBLDataset(
            data=data_train,
            train=True,
            transform=transforms.ToTensor(),
        )
        valid_dataset = MBLDataset(
            data=data_valid,
            train=False,
            transform=transforms.ToTensor(),
        )
        print("Number of training samples:", len(train_dataset))
        print("Number of validation samples:", len(valid_dataset))

        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"] = train_dataset, valid_dataset

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.hparams["batch_size"], pin_memory=True)#, num_workers=4)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):

        optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        # https://pytorch.org/docs/stable/optim.html                           #
        ########################################################################

        # Generate a list of params for the optimizer to accept:
        # params = []
        # for idx, layer_group in enumerate(self.cnnlayers):
        #     for layer in layer_group:
        #         params.append({'params': layer.parameters()})

        # for idx, layer in enumerate(self.layers):
        #         params.append({'params': layer.parameters()})

        if "use_adam" in self.hparams and self.hparams["use_adam"] == 1:
            use_adam = True
        else:
            use_adam = False

        if use_adam:
            print("Using Adam optimizer.")
            optimizer = torch.optim.Adam(self.model.parameters(), 5e-5, weight_decay=self.hparams["weight_decay"])# self.hparams["learning_rate"])
            scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
            return [optimizer], [scheduler1, scheduler2]
        else:
            print("Using Adadelta optimizer.")
            optimizer = torch.optim.Adadelta(self.model.parameters(), weight_decay=self.hparams["weight_decay"])
            return optimizer

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

