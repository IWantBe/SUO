import torch
import torch.nn as nn
import torch.nn.functional as F
import losses


class SUO(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha=0.5,
        beta=0.7,
        gamma=0.5,
    ):
        super().__init__()

        self.num_classes = out_features
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 21, 1, 'same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, 21, 1, 'same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 21, 1, 'same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 21, 1, 'same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 21, 1, 'same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 21, 1, 'same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        self.fc1 = nn.Linear(in_features // 4 * 32, 256)
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, out_features),
        )

        self.init_weights()

    def init_weights(self):
        '''Weight initialization'''
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_src, x_tar, y_src):
        # Extract features
        x_src = self.conv(x_src)
        x_tar = self.conv(x_tar)

        # Distribution alignment
        x_src_mmd1 = self.fc1(x_src)
        x_tar_mmd1 = self.fc1(x_tar)

        # Categorize the results
        y_src_predict = self.fc2(x_src_mmd1)
        y_tar_predict = self.fc2(x_tar_mmd1)

        loss_cls = F.cross_entropy(y_src_predict, y_src)
        loss_lmmd = losses.lmmd_loss(x_src_mmd1, x_tar_mmd1, y_src, y_tar_predict.softmax(1), class_num=self.num_classes)
        loss_pseudo = F.cross_entropy(y_tar_predict, y_tar_predict.argmax(1))
        loss_mcc = losses.minimum_class_confusion_loss(y_tar_predict)
        loss = loss_cls + self.alpha * loss_lmmd + self.beta * loss_pseudo + self.gamma * loss_mcc

        return loss

    def predict(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
