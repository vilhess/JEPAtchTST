import torch
import torch.nn as nn
import lightning as L

class MainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MainBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SimpleBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        return self.bn(self.conv(x))
    
class TimeResNet(nn.Module):
    def __init__(self, config):
        super(TimeResNet, self).__init__()

        win_size = config.ws
        in_channels = config.in_dim
        num_classes = config.n_classes
        feature_map = 64

        self.conv_x1 = MainBlock(in_channels, feature_map, kernel_size=8)
        self.conv_y1 = MainBlock(feature_map, feature_map, kernel_size=5)
        self.conv_z1 = SimpleBlock(feature_map, feature_map, kernel_size=3)

        self.shortcut_y1 = nn.Sequential(
            nn.Conv1d(in_channels, feature_map, kernel_size=1, stride=1, padding="same"),
            nn.BatchNorm1d(feature_map)
        )

        self.conv_x2 = MainBlock(feature_map, feature_map*2, kernel_size=8)
        self.conv_y2 = MainBlock(feature_map*2, feature_map*2, kernel_size=5)
        self.conv_z2 = SimpleBlock(feature_map*2, feature_map*2, kernel_size=3)

        self.shortcut_y2 = nn.Sequential(
            nn.Conv1d(feature_map, feature_map*2, kernel_size=1, stride=1, padding="same"),
            nn.BatchNorm1d(feature_map*2)
        )

        self.conv_x3 = MainBlock(feature_map*2, feature_map*2, kernel_size=8)
        self.conv_y3 = MainBlock(feature_map*2, feature_map*2, kernel_size=5)
        self.conv_z3 = SimpleBlock(feature_map*2, feature_map*2, kernel_size=3)

        self.shortcut_y3 = nn.Sequential(
            nn.Conv1d(feature_map*2, feature_map*2, kernel_size=1, stride=1, padding="same"),
            nn.BatchNorm1d(feature_map*2)
        )
        self.global_avg_pool = nn.AvgPool1d(1)

        self.fc = nn.Linear(feature_map*2*win_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):

        input = input.permute(0, 2, 1)

        x = self.conv_x1(input)
        x = self.conv_y1(x)
        x = self.conv_z1(x)

        out_block_1 = self.relu(x + self.shortcut_y1(input))

        x = self.conv_x2(out_block_1)
        x = self.conv_y2(x)
        x = self.conv_z2(x)

        out_block_2 = self.relu(x + self.shortcut_y2(out_block_1))

        x = self.conv_x3(out_block_2)
        x = self.conv_y3(x)
        x = self.conv_z3(x)

        out_block_3 = self.relu(x + self.shortcut_y3(out_block_2))

        x = self.global_avg_pool(out_block_3)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        
        return x
    
class ResNetLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = TimeResNet(config)
        self.lr = config.lr
        self.criterion = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        loss = self.criterion(prediction, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer