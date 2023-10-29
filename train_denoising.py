import torch
import argparse
from data.dataUtils import DataUtils
from model.ImageDenoisingModel import ImageDenoisingModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.utils import Utils
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
print("pytorch version: ", torch.__version__)

# start of argument parser
parser = argparse.ArgumentParser(description="AutoEncoder Model for image reconstruction")
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default : 1')
parser.add_argument('--train', action='store_true', help='training an AutoEncoder model')
parser.add_argument('--evaluate', action='store_true', help='evaluating a trained CNN model by a tensor')
parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR',
                    help='learning rate for training (default : 0.01)')
parser.add_argument('--outf', default='/output', help='folder for output images and model checkpoints')
parser.add_argument('--ckpf', default='', help='path to model checkpoint file to continue training')

args = parser.parse_args()
lr = args.learning_rate

# Is CUDA available?
cuda = torch.cuda.is_available()
print("Is cuda available: ", cuda)
# Seed for replication
torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)


def train_model(model, train_loader, validation_loader, optimizer, image_size, n_epochs=5, noise_factor=0.2):
    # global variable
    N_test = len(DataUtils.get_validate_dataset(image_size))
    print(" Number of validate images: ", N_test)

    loss_list = []
    criterion = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        print(" starting epoch number {}/{} ....".format(epoch, n_epochs))

        for x, label in train_loader:
            # create noise data pair for the train data
            data = x + torch.randn_like(x) * noise_factor
            data = torch.clip(data, min=0.0, max=1.0)
            if cuda:
                x, data = x.cuda(), data.cuda()
                model = model.cuda()

            # call train() on model which extents nn.module
            model.train()
            # reset the weights derivative values
            optimizer.zero_grad()
            # predict the output
            pred = model(data)
            # calculate Mean Squared loss
            loss = criterion(pred, x)
            # Calculate derivative of loss w.r.t weights
            loss.backward()
            # update the weights value
            optimizer.step()

            loss_list.append(loss.data)

    return model, loss_list


if args.train:
    latent_dim = 64
    SIZE = 28
    batch_size = 100
    epochs = 10
    channels = 1
    noise_factor = 0.2

    model = ImageDenoisingModel(image_size=SIZE,channels=channels)
    optimizer = torch.optim.Adam(model.parameters())
    train_dataset = DataLoader(dataset=DataUtils.get_train_dataset(SIZE), batch_size=batch_size)
    val_dataset = DataLoader(dataset=DataUtils.get_validate_dataset(SIZE), batch_size=batch_size * 50)

    Utils.showData(DataUtils.get_train_dataset(SIZE)[3], SIZE)
    Utils.showNoisyData(DataUtils.get_train_dataset(SIZE)[3], image_size=SIZE, noise_factor=noise_factor)
    Utils.saveModelSummary(model, channels, SIZE)

    print("Before Training starts: ")
    trained_model, LOSS = train_model(model=model, train_loader=train_dataset, validation_loader=val_dataset,
                       optimizer=optimizer, image_size=SIZE, noise_factor=noise_factor)
    print("After Training ends: ")
    # Plot out the Loss and iteration diagram
    plt.plot(LOSS)
    plt.xlabel("batch iterations ")
    plt.ylabel("Cost/total loss ")
    plt.show()

    # predict on the new data
    validate_data = DataUtils.get_validate_dataset(SIZE)
    Utils.showData(validate_data[5], SIZE)
    data_point = Utils.showNoisyData(validate_data[5], SIZE, noise_factor=noise_factor)

    trained_model.eval()
    output = trained_model(data_point)
    Utils.showDataPredicted(output.detach(), SIZE)