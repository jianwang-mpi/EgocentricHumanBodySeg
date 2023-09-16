import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, input_features, output_features=2):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_features, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Dropout2d(),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Dropout2d(),
            nn.LeakyReLU()
        )
        self.linear = nn.Linear(4096, output_features)

    def forward(self, layer3_mid_out, layer4_mid_out):
        x = torch.cat([layer3_mid_out, layer4_mid_out], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    dis = Discriminator(1024 + 2048)
    sample1 = torch.zeros([4, 1024, 16, 16])
    sample2 = torch.zeros([4, 2048, 16, 16])

    out = dis(sample1, sample2)
    print(out.shape)
