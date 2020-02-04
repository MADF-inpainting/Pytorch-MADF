import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class ProgressiveLoss(nn.Module):
    def __init__(self, extractor, args):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.args = args

    #def forward(self, input, mask, output1, output2, gt):
    def forward(self, input, mask, output1, output2, gt, feat_output_comp, feat_output, feat_gt):
        loss_dict = {}
        output_comp1 = mask * input + (1 - mask) * output1
        output_comp2 = mask * input + (1 - mask) * output2

        loss_l1_1 = 6 * self.l1((1 - mask) * output1, (1 - mask) * gt)
        loss_l1_1 += self.l1(mask * output1, mask * gt)
        loss_l1_2 = 6 * self.l1((1 - mask) * output2, (1 - mask) * gt)
        loss_l1_2 += self.l1(mask * output2, mask * gt)

        #feat_output_comp = self.extractor(output_comp2)
        #feat_output = self.extractor(output2)
        #feat_gt = self.extractor(gt)

        loss_prc = 0.0
        for i in range(3):
            loss_prc += self.l1(feat_output[i], feat_gt[i])
            loss_prc += self.l1(feat_output_comp[i], feat_gt[i])
        loss_prc *= 0.05
        return loss_l1_1 + loss_l1_2 + loss_prc
    
class InpaintingLoss(nn.Module):
    def __init__(self, extractor, args):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.args = args

    #def forward(self, input, mask, output, gt, feat_output_comp, feat_output, feat_gt):
    def forward(self, input, mask, outputs, gt, comp_feats, feats, feat_gt):
    
        loss_dict = {}
        #output_comp = mask * input + (1 - mask) * output
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
            output_comp = mask * input + (1 - mask) * outputs[i] 
            feat_output = feats[i - start]
            feat_output_comp = comp_feats[i - start]
            for j in range(3):
                loss_dict['style'] += self.l1(gram_matrix(feat_output[j]),
                                              gram_matrix(feat_gt[j]))
                loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[j]),
                                                  gram_matrix(feat_gt[j]))

            loss_dict['tv'] += total_variation_loss(output_comp)
            #if len(outputs) == 1:
            #    feat_output = feats[0]
            #    feat_output_comp = comp_feats[0]
            #    output_comp = mask * input + (1 - mask) * outputs[0] 
            #    for j in range(3):
            #        loss_dict['prc'] += self.l1(feat_output[j], feat_gt[j])
            #        loss_dict['prc'] += self.l1(feat_output_comp[j], feat_gt[j])
            #        loss_dict['style'] += self.l1(gram_matrix(feat_output[j]),
            #                                      gram_matrix(feat_gt[j]))
            #        loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[j]),
            #                                          gram_matrix(feat_gt[j]))
            #    loss_dict['tv'] = total_variation_loss(output_comp)
            #if len(outputs) == 2:
            #    feat_output = feats[1]
            #    feat_output_comp = comp_feats[1]
            #    output_comp = mask * input + (1 - mask) * outputs[1] 
            #    for j in range(3):
            #        loss_dict['style'] += self.l1(gram_matrix(feat_output[j]),
            #                                      gram_matrix(feat_gt[j]))
            #        loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[j]),
            #                                          gram_matrix(feat_gt[j]))
            #    loss_dict['tv'] = total_variation_loss(output_comp)
        #else:
        #    feat_output = feats[-1]
        #    feat_output_comp = comp_feats[-1]
        #    output_comp = mask * input + (1 - mask) * outputs[-1] 
        #    for j in range(3):
        #        loss_dict['prc'] += self.l1(feat_output[j], feat_gt[j])
        #        loss_dict['prc'] += self.l1(feat_output_comp[j], feat_gt[j])
        #        loss_dict['style'] += self.l1(gram_matrix(feat_output[j]),
        #                                      gram_matrix(feat_gt[j]))
        #        loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[j]),
        #                                          gram_matrix(feat_gt[j]))
        #    loss_dict['tv'] = total_variation_loss(output_comp)

        #loss_dict['tv'] = total_variation_loss(output_comp, mask)

        return loss_dict

