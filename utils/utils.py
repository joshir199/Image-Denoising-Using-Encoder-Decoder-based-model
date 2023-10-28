import matplotlib.pyplot as plt
from torchsummary import summary


class Utils:

    @staticmethod
    def showData(data_point, image_size):
        plt.imshow(data_point[0].numpy().reshape(image_size, image_size), cmap='gray')
        plt.title('y = ' + str(data_point[1]))
        plt.show()


    @staticmethod
    def saveModelSummary(model):
        summary(model, input_size=(1, 28, 28))
