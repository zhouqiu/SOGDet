import torch
# from mmcv.runner import force_fp32
# import torch.nn.functional as F

from mmdet.models import DETECTORS
# from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models import builder
from .bevdet import BEVDepth4D

@DETECTORS.register_module()
class BEVDepth4DOccu(BEVDepth4D):

    def __init__(self, devoxel_backbone=None,
                 devoxel_neck=None,
                 voxel_bev_head=None,
                 sequential_occ=False,
                 multi_scale=False,
                 reverse=False,
                 nocommu=False,
                 **kwargs):
        super(BEVDepth4DOccu, self).__init__(**kwargs)

        self.devoxel_neck = builder.build_neck(devoxel_neck)
        self.devoxel_backbone = builder.build_backbone(devoxel_backbone)
        self.voxel_bev_head = builder.build_head(voxel_bev_head)
        self.sequential_occ = sequential_occ
        self.multi_scale = multi_scale
        self.reverse = reverse
        self.nocommu = nocommu

    def bev_encoder(self, x_fused, x_voxel):
        if self.nocommu:
            feat = self.img_bev_encoder_backbone(x_fused)
            feat_voxel = self.devoxel_backbone(x_voxel)

            feat = self.img_bev_encoder_neck(feat)
            feat_voxel = self.devoxel_neck(feat_voxel)
            return feat, feat_voxel

        if self.reverse:
            feat = self.img_bev_encoder_backbone(x_fused)
            feat_voxel = self.devoxel_backbone(x_voxel)

            feat = self.img_bev_encoder_neck(feat)
            ms_feat_voxel = self.devoxel_neck(feat_voxel,feat)
            return feat[-1], ms_feat_voxel
        
        feat = self.img_bev_encoder_backbone(x_fused)
        feat_voxel = self.devoxel_backbone(x_voxel)
        ms_feat_voxel = self.devoxel_neck(feat_voxel)
        feat = self.img_bev_encoder_neck(feat, ms_feat_voxel)
        if self.multi_scale:
            return feat, ms_feat_voxel
        return feat, ms_feat_voxel[-1]

    def forward_voxel_bev_train(self, feats, targets):
        out = self.voxel_bev_head(feats)
        losses = self.voxel_bev_head.loss(out, targets)
        return losses

    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, depth_gt = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        depth_digit_list = []
        for img, _, _, intrin, post_rot, post_tran in zip(imgs, rots, trans,
                                                          intrins, post_rots,
                                                          post_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            # BEVDepth
            img_feat = self.img_view_transformer.featnet(x)
            depth_feat = x
            cam_params = torch.cat([intrin.reshape(B * N, -1),
                                   post_rot.reshape(B * N, -1),
                                   post_tran.reshape(B * N, -1),
                                   rot.reshape(B * N, -1),
                                   tran.reshape(B * N, -1)], dim=1)
            depth_feat = self.img_view_transformer.se(depth_feat,
                                                      cam_params)
            depth_feat = self.img_view_transformer.extra_depthnet(depth_feat)[0]
            depth_feat = self.img_view_transformer.dcn(depth_feat)
            depth_digit = self.img_view_transformer.depthnet(depth_feat)
            depth = self.img_view_transformer.get_depth_dist(depth_digit)
            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans,
                                 self.img_view_transformer.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin,
                                                          post_rot, post_tran)
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            bev_feat_list.append(bev_feat)
            depth_digit_list.append(depth_digit)

        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans,
                                              rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        if self.sequential_occ:
            x, voxel_feat = self.bev_encoder(bev_feat, bev_feat)
        else:
            x, voxel_feat = self.bev_encoder(bev_feat, bev_feat_list[0])
        return [x], depth_digit_list[0], voxel_feat

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats, depth, voxel_feat = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (img_feats, pts_feats, depth, voxel_feat)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_voxel_bev=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth, voxel_feat = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox

        depth_gt = img_inputs[-1]
        B,N,H,W = depth_gt.shape
        depth_gt = torch.split(depth_gt.view(B,2,N//2,H,W), 1, 1)[0].squeeze(1)
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        losses_voxel_bev = self.forward_voxel_bev_train(
            voxel_feat, gt_voxel_bev)
        losses.update(losses_voxel_bev)
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False, return_vox_results=False, gt_voxel_bev=None): #
        """Test function without augmentaiton."""
        img_feats, _, _, voxel_feat = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        if return_vox_results:
            voxel_out = self.voxel_bev_head(voxel_feat)
            return bbox_list, voxel_out, gt_voxel_bev
        return bbox_list

@DETECTORS.register_module()
class BEVDepth4DOccuCross(BEVDepth4D):

    def __init__(self, devoxel_backbone=None,
                 voxel_bev_head=None,
                 sequential_occ=False,
                 **kwargs):
        super(BEVDepth4DOccuCross, self).__init__(**kwargs)

        self.devoxel_backbone = builder.build_backbone(devoxel_backbone)
        self.voxel_bev_head = builder.build_head(voxel_bev_head)
        self.sequential_occ = sequential_occ
        

    def bev_encoder(self, x_fused, x_voxel):
        feat = self.img_bev_encoder_backbone(x_fused)
        feat_voxel = self.devoxel_backbone(x_voxel)
        feat, voxel_feat = self.img_bev_encoder_neck(feat, feat_voxel)
        return feat, voxel_feat

    def forward_voxel_bev_train(self, feats, targets):
        out = self.voxel_bev_head(feats)
        losses = self.voxel_bev_head.loss(out, targets)
        return losses

    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, depth_gt = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        depth_digit_list = []
        for img, _, _, intrin, post_rot, post_tran in zip(imgs, rots, trans,
                                                          intrins, post_rots,
                                                          post_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            # BEVDepth
            img_feat = self.img_view_transformer.featnet(x)
            depth_feat = x
            cam_params = torch.cat([intrin.reshape(B * N, -1),
                                   post_rot.reshape(B * N, -1),
                                   post_tran.reshape(B * N, -1),
                                   rot.reshape(B * N, -1),
                                   tran.reshape(B * N, -1)], dim=1)
            depth_feat = self.img_view_transformer.se(depth_feat,
                                                      cam_params)
            depth_feat = self.img_view_transformer.extra_depthnet(depth_feat)[0]
            depth_feat = self.img_view_transformer.dcn(depth_feat)
            depth_digit = self.img_view_transformer.depthnet(depth_feat)
            depth = self.img_view_transformer.get_depth_dist(depth_digit)
            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans,
                                 self.img_view_transformer.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin,
                                                          post_rot, post_tran)
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            bev_feat_list.append(bev_feat)
            depth_digit_list.append(depth_digit)

        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        if self.shift:
            bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans,
                                                rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        if self.sequential_occ:
            x, voxel_feat = self.bev_encoder(bev_feat, bev_feat)
        else:
            x, voxel_feat = self.bev_encoder(bev_feat, bev_feat_list[0])
        return [x], depth_digit_list[0], voxel_feat

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats, depth, voxel_feat = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (img_feats, pts_feats, depth, voxel_feat)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_voxel_bev=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth, voxel_feat = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox

        depth_gt = img_inputs[-1]
        B,N,H,W = depth_gt.shape
        depth_gt = torch.split(depth_gt.view(B,2,N//2,H,W), 1, 1)[0].squeeze(1)
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        losses_voxel_bev = self.forward_voxel_bev_train(
            voxel_feat, gt_voxel_bev)
        losses.update(losses_voxel_bev)
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False, return_vox_results=False, gt_voxel_bev=None): #
        """Test function without augmentaiton."""
        img_feats, _, _, voxel_feat = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        if return_vox_results:
            voxel_out = self.voxel_bev_head(voxel_feat)
            return bbox_list, voxel_out, gt_voxel_bev
        return bbox_list