from torchvision.transforms import v2
import torch
import torchvision

class DataUtils:

    @staticmethod
    def getTransformFn(image_size):
        """
        :param image_size: image size (W == H)
        :return: transform function for training
        """
        compose = v2.Compose([v2.RandomHorizontalFlip(),
                              v2.Resize((image_size, image_size)),
                              v2.ToImage(),
                              v2.ToDtype(torch.float32, scale=True)])
        return compose

    @staticmethod
    def get_train_dataset(image_size):
        """
        :param image_size: image size (W == H)
        :return: return training dataset after applying transform
        """
        transform = DataUtils.getTransformFn(image_size)
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                          download=True,
                                                          transform=transform)
        return train_dataset

    @staticmethod
    def get_validate_dataset(image_size):
        """
        :param image_size: image size (W == H)
        :return: return validate dataset after applying transform
        """
        compose = v2.Compose([v2.Resize((image_size, image_size)),
                              v2.ToImage(),
                              v2.ToDtype(torch.float32, scale=True)])
        validate_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                             download=True,
                                                             transform=compose)
        return validate_dataset