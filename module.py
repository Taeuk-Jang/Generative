import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from sklearn.cluster import KMeans
from torch.autograd import Variable
import torch.nn.functional as F

from utils import init_weights
from scipy.optimize import linear_sum_assignment
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self, input_dim = 512, hidden_dim = 256, output_dim = 1):
        super(Classifier, self).__init__()
        self.input_dim = input_dim 
        self.dense1 = nn.Linear(input_dim, hidden_dim)
#         self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, output_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.leakyrelu(self.dense1(x))
#         x = self.leakyrelu(self.dense2(x))
        x = self.dense3(x)
        
        return x
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]
    
class Encoder(nn.Module):
    def __init__(self, input_dim = 1, input_shape = 32):
        super(Encoder, self).__init__()
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(int(input_shape/4 * input_shape/4 * 16), 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc21 = nn.Linear(512, 64)
        self.fc22 = nn.Linear(512, 64)

        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        # W X H
        conv1 = self.relu(self.bn1(self.conv1(x)))
        # W/2 X H/2
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        # W/2 X H/2
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        # W/4 X H/4
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, int(self.input_shape/4 * self.input_shape/4 * 16))

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        mu, logvar = self.fc21(fc1), self.fc22(fc1)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]
   
    
class Encoder_img(nn.Module):
    def __init__(self, nc = 3, ndf = 64, latent_variable_size = 1024):
        super(Encoder_img, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 7, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 7, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 7, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 7, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 7, 2, 0)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*3*3, latent_variable_size)
        self.fc_bn1 = nn.BatchNorm1d(latent_variable_size)
        self.fc2 = nn.BatchNorm1d(latent_variable_size)
  
        self.fc21 = nn.Linear(latent_variable_size, latent_variable_size)
        self.fc22 = nn.Linear(latent_variable_size, latent_variable_size)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*3*3)
    
        fc1 = self.relu(self.fc_bn1(self.fc1(h5)))
        fc2 = self.relu(self.fc2(fc1))
        
        mu, logvar = self.fc21(fc2), self.fc22(fc2)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]
    
    
class Decoder_img(nn.Module):
    def __init__(self, nc = 3, ngf = 64, latent_variable_size = 1024):
        super(Decoder_img, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.latent_variable_size = latent_variable_size

        # decoder
        self.dense1 = nn.Linear(latent_variable_size, ngf*8*2*3*3)

        self.d1 = nn.ConvTranspose2d(ngf*8*2,
                                       ngf*8,
                                       kernel_size=7,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1)
        self.bn1 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.d2 = nn.ConvTranspose2d(ngf*8,
                               ngf*4,
                               kernel_size=7,
                               stride = 2,
                               padding=1,
                               output_padding=1)
        self.bn2 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.d3 = nn.ConvTranspose2d(ngf*4,
                                       ngf*2,
                                       kernel_size=7,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1)
        self.bn3 = nn.BatchNorm2d(ngf*2, 1.e-3)
        
        self.d4 = nn.ConvTranspose2d(ngf*2,
                                       ngf,
                                       kernel_size=7,
                                       stride = 2,
                                       padding=0,
                                       output_padding=0)
        self.bn4 = nn.BatchNorm2d(ngf, 1.e-3)
        
        self.d5 = nn.ConvTranspose2d(ngf,
                                       ngf,
                                       kernel_size=7,
                                       stride = 2,
                                       padding=0,
                                       output_padding=1)
        self.bn5 = nn.BatchNorm2d(ngf, 1.e-3)

        self.d6 = nn.Conv2d(ngf, out_channels= 3,
                                    kernel_size= 3, padding= 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.dense1(z).view(-1, self.ngf*8*2, 3, 3)
        h1 = self.leakyrelu(self.bn1(self.d1(z)))
        h2 = self.leakyrelu(self.bn2(self.d2(h1)))
        h3 = self.leakyrelu(self.bn3(self.d3(h2)))
        h4 = self.leakyrelu(self.bn4(self.d4(h3)))
        h5 = self.leakyrelu(self.bn5(self.d5(h4)))
        

        return self.sigmoid(self.d6(h5))
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]    
    
    
    
class _Residual_Block(nn.Module): 
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()
        
        midc=int(outc*scale)
        
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
          self.conv_expand = None
          
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x): 
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output,identity_data)))
        return output 

class Encoder_Res(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Encoder_Res, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        self.hdim = hdim
        cc = channels[0]
        self.main = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc = nn.Linear((cc)*4*4, 2*hdim)           
    
    def forward(self, x):        
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]   
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 
        
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)
    
class Decoder_Res(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Decoder_Res, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        cc = channels[-1]
        self.fc = nn.Sequential(
                      nn.Linear(hdim, cc*4*4),
                      nn.ReLU(True),
                  )
                  
        sz = 4
        
        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz*2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz*2
       
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))
                    
    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.main(y)
        return y
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]    
    
    
class SD_VAE(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(SD_VAE, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        self.hdim = hdim
        cc = channels[0]

        #Encoder Part
        self.encoder_i = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.encoder_i.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.encoder_i.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.encoder_i.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc1_i = nn.Linear((cc)*4*4, 2*hdim)

        self.encoder_r = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.encoder_r.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.encoder_r.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.encoder_r.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc1_r = nn.Linear((cc)*4*4, 2*hdim)
        
        #Projector Part
        self.projector = nn.Sequential(
            nn.Linear(2 * hdim, 2 * hdim),
            nn.ReLU(),
            nn.Linear(2 * hdim, 2 * hdim),
            nn.ReLU(),
            nn.Linear(2 * hdim, 2 * hdim)
        )
        
        #Decoder Part
        cc = channels[-1]
        self.fc2 = nn.Sequential(
                      nn.Linear(hdim * 2, cc*4*4),
                      nn.ReLU(True),
                  )
                  
        sz = 4
        self.decoder = nn.Sequential()
        for ch in channels[::-1]:
            self.decoder.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.decoder.add_module('up_to_{}'.format(sz*2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz*2
       
        self.decoder.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.decoder.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))    
        
    def encode(self, x):        
        y_i = self.encoder_i(x).view(x.size(0), -1)
        y_i = self.fc1_i(y)
        mu_i, logvar_i = y_i.chunk(2, dim=1)
        z_i = self.reparameterize(mu_i, logvar_i)
        
        y_r = self.encoder_r(x).view(x.size(0), -1)
        y_r = self.fc1_r(y)
        mu_r, logvar_r = y_r.chunk(2, dim=1)
        z_r = self.reparameterize(mu_r, logvar_r)
        
        return z_i, mu_i, logvar_i, z_r, mu_r, logvar_r
    
    def project(self, z):
        z = projector(z)
        return z
    
    def decode(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc2(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.main(y)
        return y
    
    def forward(self, x):
        z_i, mu_i, logvar_i, z_r, mu_r, logvar_r = self.encoder(x)
        
        z = torch.cat([z_i, z_r], dim = 1)
        z = self.project(z)
        recon = self.decode(z)
        
        return recon, z_i, mu_i, logvar_i, z_r, mu_r, logvar_r
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]   
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 
        
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)
    
class SD_VAE(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(SD_VAE, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        self.hdim = hdim
        cc = channels[0]

        #Encoder Part
        self.encoder_i = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.encoder_i.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.encoder_i.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.encoder_i.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc1_i = nn.Linear((cc)*4*4, 2*hdim)

        self.encoder_r = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.encoder_r.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.encoder_r.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.encoder_r.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc1_r = nn.Linear((cc)*4*4, 2*hdim)
        
        #Projector Part
        self.projector = nn.Sequential(
            nn.Linear(2 * hdim, 2 * hdim),
            nn.ReLU(),
            nn.Linear(2 * hdim, 2 * hdim),
        )
        
        #Decoder Part
        cc = channels[-1]
        self.fc2 = nn.Sequential(
                      nn.Linear(hdim * 2, cc*4*4),
                      nn.ReLU(True),
                  )
                  
        sz = 4
        self.decoder = nn.Sequential()
        for ch in channels[::-1]:
            self.decoder.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.decoder.add_module('up_to_{}'.format(sz*2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz*2
       
        self.decoder.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.decoder.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))    
        
    def encode(self, x):        
        y_i = self.encoder_i(x).view(x.size(0), -1)
        y_i = self.fc1_i(y_i)
        mu_i, logvar_i = y_i.chunk(2, dim=1)
        z_i = self.reparameterize(mu_i, logvar_i)
        
        y_r = self.encoder_r(x).view(x.size(0), -1)
        y_r = self.fc1_r(y_r)
        mu_r, logvar_r = y_r.chunk(2, dim=1)
        z_r = self.reparameterize(mu_r, logvar_r)
        
        return z_i, mu_i, logvar_i, z_r, mu_r, logvar_r
    
    def project(self, z):
        z = self.projector(z)
        return z
    
    def decode(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc2(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.decoder(y)
        return y
    
    def forward(self, x):
        z_i, mu_i, logvar_i, z_r, mu_r, logvar_r = self.encode(x)
        z = torch.cat([z_i, z_r], dim = 1)
        z = self.project(z)
        recon = self.decode(z)
        
        return recon, z_i, mu_i, logvar_i, z_r, mu_r, logvar_r
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]   
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 
        
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)
    
class _Residual_Block(nn.Module): 
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()
        
        midc=int(outc*scale)
        
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
          self.conv_expand = None
          
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x): 
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
#         output = self.relu2(self.bn2(torch.add(output,identity_data)))
        return output 
