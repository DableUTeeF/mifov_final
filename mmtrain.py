import sys
# sys.path.extend(['/home/palm/PycharmProjects/mmdetection'])
import os.path as osp
from mmdet.apis import set_random_seed
from decimal import Decimal
import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


@DATASETS.register_module()
class AlgeaDataset(CustomDataset):
    CLASSES = ('mif', 'ov')

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if 'ann_info' in data:
            data['ann_info']['bboxes'] = np.array(data['ann_info']['bboxes'], dtype='float32')
            data['ann_info']['labels'] = np.array(data['ann_info']['labels'], dtype='int64')
        return data


if __name__ == '__main__':
    # dataset = AlgeaDataset('/home/palm/PycharmProjects/mmdetection/anns/train.json', [])
    # for i in range(len(dataset)):
    #     x = dataset[0]

    cfg = Config.fromfile('/home/palm/PycharmProjects/mmdetection/configs/detr/detr_r50_8x4_150e_algea.py')  # YOLO, Faster R-CNN
    # cfg = Config.fromfile('/home/palm/PycharmProjects/mmdetection/configs/retinanet/retinanet_algea.py')  # YOLO, Faster R-CNN

    # Modify dataset type and path
    cfg.dataset_type = 'AlgeaDataset'
    cfg.data_root = ''

    cfg.data.train.type = 'AlgeaDataset'
    cfg.data.train.data_root = ''
    cfg.data.train.ann_file = '/home/palm/PycharmProjects/mmdetection/anns/train.json'
    cfg.data.train.img_prefix = ''

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = '/home/palm/PycharmProjects/mmdetection/checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
    # cfg.load_from = '/home/palm/PycharmProjects/mmdetection/checkpoints/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
    # cfg.resume_from = 'detr_lab_2/epoch_12.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = '/media/palm/BiggerData/algea/weights/detr_field_1'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = cfg.optimizer.lr / 4
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10
    cfg.lr_config.step = [10, 20]

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = [0]

    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 1
    cfg.total_epochs = 30

    # cfg.data.workers_per_gpu = 0

    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=False)
