
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import BEVdataset
import posenet

root = "dataset/CityUHK-X-BEV-master/CityUHK-X-BEV"

train_key = 'train'
test_key = 'test'
all_key = 'all'

valid_ratio = 0.2
keys = ['image', 'head_map', 'feet_map', 'bev_map', 'camera_height', 'camera_angle', 'camera_fu', 'camera_fv']
use_augment = True

datalist = BEVdataset.load_datalist(root, True)
# print(datalist[train_key])
num_train = len(datalist[train_key]) * (1 - valid_ratio)
num_train = int(num_train)

train_datalist = datalist[train_key][:num_train]
valid_datalist = datalist[train_key][num_train:]
test_datalist = datalist[test_key]

train_dataset = BEVdataset.CityUHKBEV(root, train_datalist, keys, use_augment=True)
valid_dataset = BEVdataset.CityUHKBEV(root, valid_datalist, keys, use_augment=False)
test_dataset = BEVdataset.CityUHKBEV(root, test_datalist, keys, use_augment=False)

train_dataset = DataLoader(train_dataset, batch_size=64, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = posenet.PoseNet("resnet101", True).to(device)


def train(dataloader):
#     model.train()
    backward = True
    loss_value = 0
    for batch in dataloader:

        image = batch['image'].to(device)
        camera_height = batch['camera_height'].to(device)
        camera_angle = batch['camera_angle'].to(device)

        pred = model(image)

        loss = loss_fn(pred['camera_height'], pred['camera_angle'], camera_height, camera_angle)
        if backward:
            optimizer.zero_grad()
            loss['pose'].backward()
            optimizer.step()

        # batch.pred = pred
        # batch.loss = loss
        # batch.size = image.size(0)



        # if tmp > 0:
        #     if (tmp % 10) == 0:
        #         torch.save(model.state_dict(), './model_checkpoint.chp')

    return loss['pose'].item()

	
class PoseLoss(nn.Module):
    def __init__(self, height_loss_weight=0.02):
        super(PoseLoss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.height_loss_weight = height_loss_weight

    def forward(self, pred_height, pred_angle, target_height, target_angle):
        loss_height = self.loss_fn(pred_height, target_height)
        loss_angle = self.loss_fn(pred_angle, target_angle)
        loss = loss_angle + loss_height * self.height_loss_weight
        return {
            'pose': loss,
            'pose-height': loss_height,
            'pose-angle': loss_angle
        }


loss_fn = PoseLoss(0.02)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.00001)

EPOCHS = 1
for epoch in range(1, EPOCHS+1):
    train_loss = train(train_dataset)
    test_loss_, accuracy_ = 0, 0 # test()
    print(f"the {epoch}/{EPOCHS} epoch, training loss is {train_loss}, testing loss is {test_loss_}, testing accuracy is {accuracy_}")
