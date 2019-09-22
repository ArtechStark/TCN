import torch
from torchsummary import summary
import sys
sys.path.append('../')
from TCN.poly_music.model import TCN

model = TCN(input_size=100, output_size=100,
            num_channels=[2, 25, 100],kernel_size = 10,dropout=0.5).cuda()
summary(model, (100,25), batch_size=5)




