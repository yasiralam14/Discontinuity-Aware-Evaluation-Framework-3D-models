def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None):
        feat_x, feat_y = self.net(x), self.net(y)
        
        # Square differences at each scale
        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        
        res = []
        for d, l in zip(diff, self.lin):
            # 1. Compute the distance map (before averaging)
            dist_map = l(d) 
            
            if mask is not None:
                if mask is not None and mask.shape[1] > 1:
                    mask = mask[:, 0:1, :, :]
                cur_mask = F.interpolate(mask, size=dist_map.shape[-2:], mode='nearest')
                
                numerator = (dist_map * cur_mask).sum(dim=(2, 3), keepdim=True)
                denominator = cur_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                
                res.append(numerator / denominator)
            else:
                res.append(dist_map.mean(dim=(2, 3), keepdim=True))

        return  torch.cat(res, 1).sum(dim=1).mean()
