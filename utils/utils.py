import matplotlib.pyplot as plt
from torchsummary import summary
import torch


class Utils:

    @staticmethod
    def showData(data_point, image_size):
        """
        :param data_point: image data
        :param image_size: image size (W == H)
        :return: show image
        """
        plt.imshow(data_point[0].numpy().reshape(image_size, image_size), cmap='gray')
        plt.title('y = ' + str(data_point[1]))
        plt.show()


    @staticmethod
    def saveModelSummary(model, channels, image_size):
        """
        :param model: model to be trained
        :param channels: number of channel
        :param image_size: image size (W == H)
        :return: show model summary
        """
        summary(model, input_size=(channels, image_size, image_size))

    @staticmethod
    def showNoisyData(data_point, image_size, noise_factor):
        """
        :param data_point: image data
        :param image_size: image size (W == H)
        :param noise_factor: % amount of noise in input image
        :return: show noisy image and return noisy image
        """
        data = data_point[0] + torch.randn_like(data_point[0]) * noise_factor
        data = torch.clip(data, min=0.0, max=1.0)
        plt.imshow(data.numpy().reshape(image_size, image_size), cmap='gray')
        plt.show()
        return data

    @staticmethod
    def showDataPredicted(data_point, image_size):
        """
        :param data_point: image data
        :param image_size: image size (W == H)
        :return: show predicted image
        """
        plt.imshow(data_point.numpy().reshape(image_size, image_size), cmap='gray')
        plt.show()
