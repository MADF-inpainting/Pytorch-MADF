import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from normalization import PN
from torchvision import models

upsampling = "bilinear"
    
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class FilterGen(nn.Module):
    def __init__(self, in_ch, conv_in_ch, conv_out_ch, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_in_ch = conv_in_ch
        self.conv_out_ch = conv_out_ch
        nhidden = 16
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, nhidden, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(nhidden, conv_in_ch * kernel_size * kernel_size * conv_out_ch, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, input):
        mask = self.conv(input)
        filters = self.conv1(mask)
        N, C, H, W = filters.shape
        filters = filters.view((N, C, H*W))
        #filters = torch.nn.functional.unfold(filters, (1, 1), dilation=1, padding=0, stride=1)
        filters = filters.transpose(1,2)
        filters = filters.view((N, H*W, self.conv_in_ch * self.kernel_size * self.kernel_size, self.conv_out_ch))
        return filters, mask

class ConvWithFilter(nn.Module):
    def __init__(self, out_ch, kernel_size, stride=0, padding=0, dilation=1, label_ch=0, bn='batchnorm', activ='relu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.bn = bn
        self.activ=activ
        if bn=="batchnorm":
            self.bn = nn.BatchNorm2d(out_ch)
        elif bn=="PN":
            self.bn = PN(out_ch, label_ch, upsampling) 
        
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
    

    def forward(self, features, filters):
        N, C, in_H, in_W = features.shape
        out_H = (in_H - self.kernel_size + 2 * self.padding) / self.stride + 1
        out_W = (in_W - self.kernel_size + 2 * self.padding) / self.stride + 1
        out_H = int(out_H)
        out_W = int(out_W)
        features_unf = torch.nn.functional.unfold(features, (self.kernel_size, self.kernel_size), dilation=self.dilation, padding=self.padding, stride=self.stride)
        features_ = features_unf.transpose(1, 2).unsqueeze(2)
        result = features_.matmul(filters)
        result = result.squeeze(2) 
        result = result.transpose(1, 2)
        N, C, _ = result.shape 
        result = result.view(N, C, out_H, out_W)
        #result = torch.nn.functional.fold(result, (out_H, out_W), (1, 1))
        if hasattr(self, 'bn') and self.bn=="batchnorm":
            out = self.bn(result)
        elif hasattr(self, 'bn') and self.bn=="PN":
            out = self.bn(result, edge_map)
        if hasattr(self, 'activation'):
            out = self.activation(result)
        return out


class AttConv(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, activation='relu'):
        super(AttConv,self).__init__()
        if activation is False:
            self.conv = nn.Sequential(
                nn.Conv2d(query_dim+key_dim, query_dim, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(query_dim+key_dim, query_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

    def forward(self, query, key):
        out = key
        return self.conv(torch.cat([out, query], dim = 1))
        #return out


class DecActiv(nn.Module):
    def __init__(self, in_ch, out_ch, label_ch=0, bn=False, activ='relu'):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

        self.bn_type=bn
        if bn=="PN":
            self.bn = PN(out_ch, label_ch, upsampling)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, edge_map=None):
        h = self.deconv(input)
        if hasattr(self, 'bn') and self.bn_type=="PN":
            h = self.bn(h, edge_map)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h

class MADFNet(nn.Module):
    def __init__(self, args, layer_size=7, input_channels=3, upsampling_mode=upsampling):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.n_refinement_D = args.n_refinement_D
        ###Encoder
        self.filter_gen_1 = FilterGen(in_ch = 3, conv_in_ch=input_channels*2, conv_out_ch=16, kernel_size=7, stride=2, padding=3)
        self.filter_gen_2 = FilterGen(16, 16, 32, 5, 2, 2)
        self.filter_gen_3 = FilterGen(16, 32, 64, 3, 2, 1)
        self.filter_gen_4 = FilterGen(16, 64, 128, 3, 2, 1)
        for i in range(5, layer_size + 1):
            setattr(self, "filter_gen_{:d}".format(i), FilterGen(16, 128, 128, 3, 2, 1))
        #self.filter_gen_5 = FilterGen(16, 128, 128, 3, 2, 1)
        #self.filter_gen_6 = FilterGen(16, 128, 128, 3, 2, 1)
        #self.filter_gen_7 = FilterGen(16, 128, 128, 3, 2, 1)
        self.enc_conv_1 = ConvWithFilter(out_ch=16,  kernel_size=7, stride=2, padding=3) 
        self.enc_conv_2 = ConvWithFilter(32, 5, 2, 2) 
        self.enc_conv_3 = ConvWithFilter(64, 3, 2, 1)
        for i in range(4, layer_size + 1):
            setattr(self, "enc_conv_{:d}".format(i), ConvWithFilter(128, 3, 2, 1))
         
        #self.enc_conv_4 = ConvWithFilter(128, 3, 2, 1) 
        #self.enc_conv_5 = ConvWithFilter(128, 3, 2, 1) 
        #self.enc_conv_6 = ConvWithFilter(128, 3, 2, 1) 
        #self.enc_conv_7 = ConvWithFilter(128, 3, 2, 1)
        self.enc_up_1 = nn.Sequential( 
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.enc_up_2 = nn.Sequential( 
            nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.enc_up_3 = nn.Sequential( 
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        for i in range(4, layer_size + 1):
            enc_up = nn.Sequential(nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
            setattr(self, "enc_up_{:d}".format(i), enc_up)
        #######Encoder end
        #######Recovery Decoder
        for i in range(self.layer_size, 4, -1):
            name = 'deconv_{:d}'.format(i)
            dconv = []
            dconv.append(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1))
            dconv.append(nn.BatchNorm2d(512))
            dconv.append(nn.LeakyReLU(negative_slope=0.2))
            setattr(self, name, nn.Sequential(*dconv))
        self.deconv_4 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.2))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1))
        for i in range(self.layer_size, 4, -1):
             setattr(self, 'att_conv_{:d}'.format(i), AttConv(query_dim=512, key_dim=512, value_dim=512))
        self.att_conv_4 = AttConv(query_dim=256, key_dim=256, value_dim=256)
        self.att_conv_3 = AttConv(query_dim=128, key_dim=128, value_dim=128)
        self.att_conv_2 = AttConv(query_dim=64, key_dim=64, value_dim=64)
        self.att_conv_1 = AttConv(query_dim=3, key_dim=6, value_dim=6, activation=False)
        if self.n_refinement_D > 0: 
            for i in range(self.layer_size, 4, -1):
                name = 'dec_ref0_{:d}'.format(i)
                setattr(self, name, DecActiv(512, 512, label_ch=512, bn="PN", activ='leaky'))
            self.dec_ref0_4 = DecActiv(512, 256, label_ch=256, bn="PN", activ='leaky')
            self.dec_ref0_3 = DecActiv(256, 128, label_ch=128, bn="PN", activ='leaky')
            self.dec_ref0_2 = DecActiv(128, 64, label_ch=64, bn="PN", activ='leaky')
            self.dec_ref0_1 = nn.Conv2d(64 + input_channels*3, input_channels, kernel_size=1, stride=1, padding=0)
        
        if self.n_refinement_D > 1: 
            for i in range(self.layer_size, 4, -1):
                name = 'dec_ref1_{:d}'.format(i)
                setattr(self, name, DecActiv(512, 512, label_ch=512, bn="PN", activ='leaky'))
            self.dec_ref1_4 = DecActiv(512, 256, label_ch=256, bn="PN", activ='leaky')
            self.dec_ref1_3 = DecActiv(256, 128, label_ch=128, bn="PN", activ='leaky')
            self.dec_ref1_2 = DecActiv(128, 64, label_ch=64, bn="PN", activ='leaky')
            self.dec_ref1_1 = nn.Conv2d(64 + input_channels*4, input_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_dict['h_0'] = torch.cat([input, input_mask], dim=1)
        mask_pre = input_mask
        pre_conv = h_dict['h_0']
        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            h_key = 'h_{:d}'.format(i)
            filters, mask_res = getattr(self, 'filter_gen_{:d}'.format(i))(mask_pre)
            mask_pre = mask_res
            conv_res = getattr(self, "enc_conv_{:d}".format(i))(pre_conv, filters)
            h_dict[h_key] = getattr(self, "enc_up_{:d}".format(i))(conv_res)
            pre_conv = conv_res
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h = h_dict[h_key]
        
        h_att = h
        h_second = h
        outputs = []
        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)

            dconv = getattr(self, 'deconv_{:d}'.format(i))(h_att)
            h_att = getattr(self, 'att_conv_{:d}'.format(i))(dconv, h_dict[enc_h_key])
            if i != 1:
                if self.n_refinement_D > 0:
                    h_second = getattr(self, 'dec_ref0_{:d}'.format(i))(h_second, h_att)
                if self.n_refinement_D > 1:
                    h = getattr(self, 'dec_ref1_{:d}'.format(i))(h, edge_map=h_second)
            else:
                outputs.append(h_att)
                if self.n_refinement_D > 0:
                    h_second = F.interpolate(h_second, scale_factor=2, mode=upsampling)
                    h_second = torch.cat([h_second, h_att, h_dict[enc_h_key]], dim=1)
                    h_second = getattr(self, 'dec_ref0_{:d}'.format(i))(h_second)
                    outputs.append(h_second)
                if self.n_refinement_D > 1:
                    h = F.interpolate(h, scale_factor=2, mode=upsampling)
                    h = torch.cat([h, h_att, h_second, h_dict[enc_h_key]], dim=1)
                    h = getattr(self, 'dec_ref1_{:d}'.format(i))(h)
                    outputs.append(h)
        return outputs 

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        #if self.freeze_enc_bn:
        #    for name, module in self.named_modules():
        #        if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
        #            module.eval()


if __name__ == '__main__':
    size = (1, 3, 5, 5)
    input = torch.ones(size)
    input_mask = torch.ones(size)
    input_mask[:, :, 2:, :][:, :, :, 2:] = 0

    conv = PartialConv(3, 3, 3, 1, 1)
    l1 = nn.L1Loss()
    input.requires_grad = True

    output, output_mask = conv(input, input_mask)
    loss = l1(output, torch.randn(1, 3, 5, 5))
    loss.backward()

    assert (torch.sum(input.grad != input.grad).item() == 0)
    assert (torch.sum(torch.isnan(conv.input_conv.weight.grad)).item() == 0)
    assert (torch.sum(torch.isnan(conv.input_conv.bias.grad)).item() == 0)

