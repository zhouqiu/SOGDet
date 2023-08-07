import torch
import torch.distributed as dist
import numpy as np
from mmcv.runner.hooks.evaluation import DistEvalHook
import os.path as osp
import warnings
from math import inf
from typing import Callable, List, Optional

from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import mmcv
import pickle
import tempfile
from mmcv.runner import get_dist_info
from mmcv.engine import collect_results_cpu, collect_results_gpu
import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    class_names: list = None):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    CalMeanIou_vox = MeanIoU(class_indices=range(len(class_names)), names=class_names)
    CalMeanIou_vox.reset()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # result = model(return_loss=False, rescale=True, **data)
            result, voxel_out, gt_voxel_bev = model(return_loss=False, return_vox_results=True, **data) #return_vox_results=True,
            # calculate mIOU
            if len(voxel_out.shape) == len(gt_voxel_bev[0].shape): #[B, X, Y, Z]
                voxel_out = torch.round(voxel_out).int()
            else:
                voxel_out = torch.argmax(voxel_out, dim=1) #[B, C, X, Y, Z]
            for count in range(len(data["img_metas"])):
                CalMeanIou_vox._after_step(
                    voxel_out[count].flatten(),
                    gt_voxel_bev[count].flatten())
            # use out_dir
            # import pdb
            # pdb.set_trace()
            if i % 10 == 0:
                np.save(osp.join(out_dir, 'pred_{}.npy'.format(data['img_metas'][0].data[0][0]['sample_idx'])),np.array(voxel_out.cpu()))
                np.save(osp.join(out_dir, 'gt_{}.npy'.format(data['img_metas'][0].data[0][0]['sample_idx'])),np.array(gt_voxel_bev[0].cpu()))

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir=out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    CalMeanIou_vox._after_epoch()
    return results



def multi_gpu_test(model: nn.Module,
                   data_loader: DataLoader,
                   tmpdir: Optional[str] = None,
                   gpu_collect: bool = False,
                   class_names: list = None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    CalMeanIou_vox = MeanIoU(class_indices=range(len(class_names)), names=class_names)
    CalMeanIou_vox.reset()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, voxel_out, gt_voxel_bev = model(return_loss=False, return_vox_results=True, **data) #return_vox_results=True,
            # calculate mIOU
            if len(voxel_out.shape) == len(gt_voxel_bev[0].shape): #[B, X, Y, Z]
                voxel_out = torch.round(voxel_out).int()
            else:
                voxel_out = torch.argmax(voxel_out, dim=1) #[B, C, X, Y, Z]
            for count in range(len(data["img_metas"])):
                CalMeanIou_vox._after_step(
                    voxel_out[count].flatten(),
                    gt_voxel_bev[count].flatten())
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            batch_size_all = batch_size * world_size
            if batch_size_all + prog_bar.completed > len(dataset):
                batch_size_all = len(dataset) - prog_bar.completed
            for _ in range(batch_size_all):
                prog_bar.update()

    CalMeanIou_vox._after_epoch()

    # collect results from all ranks
    if gpu_collect:
        result_from_ranks = collect_results_gpu(results, len(dataset))
    else:
        result_from_ranks = collect_results_cpu(results, len(dataset), tmpdir)
    return result_from_ranks


class MeanIoU:

    def __init__(self,
                 class_indices,
                 ignore_label: int=0,
                 # label_str,
                 names: list=None,
                 # empty_class: int
                 ):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        # self.label_str = label_str
        if names is None:
            self.names = ['noise','barrier','bicycle','bus','car','construction_vehicle','motorcycle','pedestrian','traffic_cone','trailer','truck',
'driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation','empty']
        else:
            self.names = names

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def _after_step(self, outputs, targets):
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()
        # print("total_seen:{} \n total_correct:{} \n total_positive:{}".format(self.total_seen, self.total_correct, self.total_positive))

    def _after_epoch(self):
        dist.all_reduce(self.total_seen)
        dist.all_reduce(self.total_correct)
        dist.all_reduce(self.total_positive)

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou.item())

        miou = np.mean(ious)
        print(f'Validation per class iou {self.names}:')
        for iou, name in zip(ious, self.names):
            print('%s : %.2f%%' % (name, iou * 100))
        print('%s : %.2f%%' % ('mIOU', miou * 100))

        # return miou * 100

class DistODOccEvalHook(DistEvalHook):
    """Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader, whose dataset has
            implemented ``evaluate`` function.
        start (int | None, optional): Evaluation starting epoch. It enables
            evaluation before the training starts if ``start`` <= the resuming
            epoch. If None, whether to evaluate is merely decided by
            ``interval``. Default: None.
        interval (int): Evaluation interval. Default: 1.
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            default: True.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be saved in ``runner.meta['hook_msgs']`` to keep
            best score value and best checkpoint path, which will be also
            loaded when resume checkpoint. Options are the evaluation metrics
            on the test dataset. e.g., ``bbox_mAP``, ``segm_mAP`` for bbox
            detection and instance segmentation. ``AR@100`` for proposal
            recall. If ``save_best`` is ``auto``, the first key of the returned
            ``OrderedDict`` result will be used. Default: None.
        rule (str | None, optional): Comparison rule for best score. If set to
            None, it will infer a reasonable rule. Keys such as 'acc', 'top'
            .etc will be inferred by 'greater' rule. Keys contain 'loss' will
            be inferred by 'less' rule. Options are 'greater', 'less', None.
            Default: None.
        test_fn (callable, optional): test a model with samples from a
            dataloader in a multi-gpu manner, and return the test results. If
            ``None``, the default test function ``mmcv.engine.multi_gpu_test``
            will be used. (default: ``None``)
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        broadcast_bn_buffer (bool): Whether to broadcast the
            buffer(running_mean and running_var) of rank 0 to other rank
            before evaluation. Default: True.
        out_dir (str, optional): The root directory to save checkpoints. If not
            specified, `runner.work_dir` will be used by default. If specified,
            the `out_dir` will be the concatenation of `out_dir` and the last
            level directory of `runner.work_dir`.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details. Default: None.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataloader: DataLoader,
                 start: Optional[int] = None,
                 interval: int = 1,
                 by_epoch: bool = True,
                 save_best: Optional[str] = None,
                 rule: Optional[str] = None,
                 test_fn: Optional[Callable] = None,
                 greater_keys: Optional[List[str]] = None,
                 less_keys: Optional[List[str]] = None,
                 broadcast_bn_buffer: bool = True,
                 tmpdir: Optional[str] = None,
                 gpu_collect: bool = False,
                 out_dir: Optional[str] = None,
                 file_client_args: Optional[dict] = None,
                 num_classes: int = 10,
                 # class_indices: Optional[list] = None,
                 **eval_kwargs):

        if test_fn is None:
            test_fn = multi_gpu_test

        super().__init__(
            dataloader,
            start=start,
            interval=interval,
            by_epoch=by_epoch,
            save_best=save_best,
            rule=rule,
            test_fn=test_fn,
            greater_keys=greater_keys,
            less_keys=less_keys,
            out_dir=out_dir,
            file_client_args=file_client_args,
            **eval_kwargs)

        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

        self.num_classes = num_classes
        self.class_indices = range(num_classes)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.test_fn(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results) #bbox list
            # the key_score may be `None` so it needs to skip the action to
            # save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)

            #mIOU
            total_seen = torch.zeros(self.num_classes).cuda()
            total_correct = torch.zeros(self.num_classes).cuda()
            total_positive = torch.zeros(self.num_classes).cuda()

            targets = results.flatten()
            outputs = results.flatten()

            for i, c in enumerate(self.class_indices):
                total_seen[i] += torch.sum(targets == c).item()
                total_correct[i] += torch.sum((targets == c)
                                                   & (outputs == c)).item()
                total_positive[i] += torch.sum(outputs == c).item()


            dist.all_reduce(total_seen)
            dist.all_reduce(total_correct)
            dist.all_reduce(total_positive)

            ious = []

            for i in range(self.num_classes):
                if total_seen[i] == 0:
                    ious.append(1)
                else:
                    cur_iou = total_correct[i] / (total_seen[i] + total_positive[i] - total_correct[i])
                    ious.append(cur_iou.item())

            miou = np.mean(ious)
            runner.log_buffer.output['mIOU'] = miou
            runner.log_buffer.ready = True
