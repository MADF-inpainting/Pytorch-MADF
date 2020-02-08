import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

    
class InpaintingLoss(nn.Module):
    def __init__(self, extractor, args):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.args = args

    def forward(self, input, mask, outputs, gt, comp_feats, feats, feat_gt):
    
        loss_dict = {}
        loss_dict['hole'] = 0.0
        loss_dict['valid'] = 0.0
        loss_dict['prc'] = 0.0
        loss_dict['style'] = 0.0
        loss_dict['tv'] = 0.0
        if self.args.use_incremental_supervision:
            start = 0
        else:
            start = len(outputs) - 1
        for i in range(start, len(outputs)):
            print('l1 loss:', i)
            output = outputs[i]
            loss_dict['hole'] += self.l1((1 - mask) * output, (1 - mask) * gt)
            loss_dict['valid'] += self.l1(mask * output, mask * gt)

        if self.args.use_incremental_supervision:
            if len(outputs) == 1:
                start = 0
            else:
                start = 1
        else:
            start = len(outputs) - 1
             
        for i in range(start, len(outputs)):
            feat_output = feats[i - start]
            feat_output_comp = comp_feats[i - start]
            print("prc loss:", i)
            for j in range(3):
                loss_dict['prc'] += self.l1(feat_output[j], feat_gt[j])
                loss_dict['prc'] += self.l1(feat_output_comp[j], feat_gt[j])
            
        if self.args.use_incremental_supervision:
            if len(outputs) == 1:
                start = 0
            elif len(outputs) == 2:
                start = 1
            else:
                start = 2
        else:
            start = len(outputs) - 1
        
        for i in range(start, len(outputs)):
            print("style loss:", i)
            output_comp = mask * input + (1 - mask) * outputs[i] 
            feat_output = feats[i - start]
            feat_output_comp = comp_feats[i - start]
            for j in range(3):
                loss_dict['style'] += self.l1(gram_matrix(feat_output[j]),
                                              gram_matrix(feat_gt[j]))
                loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[j]),
                                                  gram_matrix(feat_gt[j]))

            loss_dict['tv'] += total_variation_loss(output_comp)
        
        return loss_dict

