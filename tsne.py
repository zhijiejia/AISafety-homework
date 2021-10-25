import os
import torch
import torchvision
from torch import nn
from tqdm import tqdm
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from net.CifarNet import CifarNet, split_weights
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tools import scheduler, losses, ext_transforms, dataset, metric, utils


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Test_transform = transforms.Compose([
   transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

testDataset = dataset.CifarDataset(train=False, transform=Test_transform)
testLoader = DataLoader(testDataset, batch_size=100, shuffle=True, num_workers=2, drop_last=True)

outputs_cls = []
outputs_labels = []

with torch.no_grad():

    model = CifarNet(num_classes=10).cuda()
    state_dict = torch.load('./best_acc_0.9422.pth')
    model.load_state_dict(state_dict)
    model.eval()
    testMetric = metric.StreamSegMetrics(n_classes=10)
    labelName = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for images, labels in tqdm(testLoader):

        images = images.to(torch.float32)
        labels = labels.to(torch.long)
        images = images.cuda()
        labels = labels.cuda()
        outputs = model.backbone(images)
        outputs_cls.append(outputs.cpu().numpy())
        outputs_labels.append(labels.cpu().numpy())

layer_output = np.vstack(outputs_cls)
layer_label = np.hstack(outputs_labels)

color=['cyan','black','green','red','blue','orange','brown','pink','purple','grey']

tsne = TSNE(n_components=2, init='pca', n_iter=5000)

emmbing_data = tsne.fit_transform(layer_output)

print(emmbing_data.shape)
print(layer_label.shape)

for index, data_point in tqdm(enumerate(emmbing_data)):
    plt.scatter(data_point[0], data_point[1], layer_label[index], c=color[layer_label[index]])
plt.savefig('./tnse.png')
