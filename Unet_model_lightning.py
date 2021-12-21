import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim

class DoubleConv(pl.LightningModule):
    def __init__(self, in_chans, out_chans):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU()
        )
    # parameters in first conv layer change the number of channels from in_chans to out_chans
    # parameters in second conv layer do not modify the size of the image

    def forward(self, x):
        return self.conv(x)




class Lit_Net(pl.LightningModule):
    def __init__(self, in_chans, out_chans):
        super().__init__()

        self.descending = nn.ModuleList()
        self.ascending = nn.ModuleList()
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)  # These parameters give back an output with half the height and width of the input
        #self.bottom = DoubleConv(512, 1024)
        self.bottom = DoubleConv(256, 512)
        self.output_layer = nn.Conv2d(64, out_chans, kernel_size=1)
        self.criterion = nn.BCEWithLogitsLoss()

        #for num_features in [64, 128, 256, 512]:
        for num_features in [64, 128, 256]:
            self.descending.append(DoubleConv(in_chans, num_features))
            in_chans = num_features
        
        #for num_features in [512, 256, 128, 64]:
        for num_features in [256, 128, 64]:
            self.ascending.append(nn.ConvTranspose2d(2*num_features, num_features, kernel_size=2, stride=2))  #these parameters double the input size
            self.ascending.append(DoubleConv(2*num_features, num_features))   # double the number of in channels again, since we concatenate with the result of descending conv layers

    
    def forward(self, x):
        # forward pass

        skip_connections = []
        i = 0
        for descending_lay in self.descending:
            i +=1
            x = descending_lay(x)
            skip_connections.append(x)
            x = self.pooling_layer(x)
        
        skip_connections = skip_connections[::-1]   # reverse the order of the skip connections list

        x = self.bottom(x)

        for idx in range(0, len(self.ascending), 2):
            x = self.ascending[idx](x)
            skip = skip_connections[idx//2]
            #if x.shape != skip:
            #  x = TF.resize(x, size=skip.shape[2, 3])
            new_input = torch.cat((skip, x), dim=1)
            x = self.ascending[idx+1](new_input)
        
        return self.output_layer(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(torch.squeeze(out, dim=1), y.type(torch.FloatTensor).cuda())
        #return {'loss': loss}
        return loss
 
