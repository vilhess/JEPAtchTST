# Random Mask Collator

import torch 

class MaskCollator(object):

    def __init__(
        self,
        ratio=(0.3, 0.5),
        window_size=100,
        patch_len=8,
    ):
        super(MaskCollator, self).__init__()

        self.patch_len= patch_len
        self.patch_num = window_size// patch_len
        self.ratio = ratio


    def __call__(self, batch):

        # Input :

        # batch: [(data1, target1), (data2, target2), ...]

        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        ratio = self.ratio 
        ratio = ratio[0] + torch.rand(1).item() * (ratio[1] - ratio[0]) # random percentage to mask 
        num_keep = int(self.patch_num * (1. - ratio))

        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(B):

            m = torch.randperm(self.patch_num)
            collated_masks_enc.append(m[:num_keep])
            collated_masks_pred.append(m[num_keep:])

        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred # bs, ws, in_dim ; bs, indices, ; bs, patch_num-indices