import torch
import torch.nn as nn
import torchvision.models as models
import os


def load_model(arch, code_length):
    """
    Load CNN model.

    Args
        arch(str): Model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier = model.classifier[:-2]
        model = ModelWrapper(model, 4096, code_length)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = model.classifier[:-3]
        model = ModelWrapper(model, 4096, code_length)
    else:
        raise ValueError("Invalid model name!")

    return model


class ModelWrapper(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, last_node, code_length):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.code_length = code_length
        self.hash_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )

        # Extract features
        self.extract_features = False

    def forward(self, x):
        if self.extract_features:
            return self.model(x)
        else:
            return self.hash_layer(self.model(x))

    def set_extract_features(self, flag):
        """
        Extract features.

        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag

    def snapshot(self, it, optimizer):
        """
        Save model snapshot.

        Args
            it(int): Iteration.
            optimizer(torch.optim): Optimizer.

        Returns
            None
        """
        torch.save({
            'iteration': it,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join('checkpoints', 'resume_{}.t'.format(it)))

    def load_snapshot(self, root, optimizer=None):
        """
        Load model snapshot.

        Args
            root(str): Path of model snapshot.
            optimizer(torch.optim): Optimizer.

        Returns
            optimizer(torch.optim, optional): Optimizer, if parameter 'optimizer' given.
            it(int): Iteration, if parameter 'optimizer' given.
        """
        checkpoint = torch.load(root)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            it = checkpoint['iteration']
            return optimizer, it
