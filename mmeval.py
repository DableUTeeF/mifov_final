"""
cascade: mAP using the weighted average of precisions among classes: 0.4544
        ov: 0.9010
        mif: 0.8108
        mAP: 0.0007

"""
import sys
# sys.path.extend(['/home/palm/PycharmProjects/mmdetection'])
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
import time
import numpy as np
import json
from evaluate_util import evaluate, all_annotation_from_instance
import os

if __name__ == '__main__':
    dataset = json.load(open('/home/palm/PycharmProjects/mmdetection/anns/test.json'))
    path = ('/home/palm/PycharmProjects/pig/mmdetection/configs/', '/media/palm/BiggerData/algea/weights/')
    configs = [
        ('cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py',
         'cascade_r101/epoch_20.pth',
         'cascade_rcnn_r101'),
        ('cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py',
         'cascade_r50/epoch_20.pth',
         'cascade_rcnn_r50'),
        ('detr/detr_r50_8x2_150e_coco.py',
         'detr_r50/epoch_30.pth',
         'detr_r50'),
        ('retinanet/retinanet_r50_fpn_1x_coco.py',
         'retinanet_r50/epoch_23.pth',
         'retinanet_r50'),
        ('retinanet/retinanet_r101_fpn_1x_coco.py',
         'retinanet_r101/epoch_24.pth',
         'retinanet_r101'),
        ('faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py',
         'rcnn_r101/epoch_30.pth',
         'rcnn_r101'),
        ('faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco.py',
         'rcnn_r50/epoch_24.pth',
         'rcnn_r50'),
        ('yolo/yolov3_d53_mstrain-608_273e_coco.py',
         'yolov3_d53/epoch_30.pth',
         'yolo_d53'),
        ('deformable_detr/deformable_detr_r50_16x2_50e_coco.py',
         'deformdetr/epoch_50.pth',
         'deformable_r50'),
        ('gfl/gfl_r50_fpn_mstrain_2x_coco.py',
         'gfl_r50/epoch_24.pth',
         'gfl_r50'),
        ('gfl/gfl_r50_fpn_mstrain_2x_coco.py',
         'gfl_r50_new/epoch_24.pth',
         'gfl_r50_new'),
        ('gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py',
         'gfl_r101_dcn/epoch_24.pth',
         'gfl_r101_dcn'),
        ('gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py',
         'gfl_x101_dcn/epoch_24.pth',
         'gfl_x101_dcn'),
        ('vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py',
         'vfnet_r50_dcn/epoch_24.pth',
         'vfnet_r50_dcn'),
        ('vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py',
         'vfnet_r50_dcn_new/epoch_24.pth',
         'vfnet_r50_dcn_new'),
        ('vfnet/vfnet_r50_fpn_mstrain_2x_coco.py',
         'vfnet_r50/epoch_24.pth',
         'vfnet_r50'),
        ('vfnet/vfnet_r50_fpn_mstrain_2x_coco.py',
         'vfnet_r50_new/epoch_24.pth',
         'vfnet_r50_new'),
        ('vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py',
         'vfnet_x101_dcn/epoch_22.pth',
         'vfnet_x101'),
    ]
    results = {}
    for config, weight, name in configs:
        config = os.path.join(path[0], config)
        weight = os.path.join(path[1], weight)
        print('\n' + name)
        cfg = Config.fromfile(config)
        if cfg.model.type == 'CascadeRCNN':
            if cfg.model.backbone.depth == 50:
                cfg.model.roi_head.bbox_head[0].num_classes = 2
                cfg.model.roi_head.bbox_head[1].num_classes = 2
                cfg.model.roi_head.bbox_head[2].num_classes = 2
        elif cfg.model.type == 'FasterRCNN':
            cfg.model.roi_head.bbox_head.num_classes = 2
        else:
            cfg.model.bbox_head.num_classes = 2

        # Build the detector
        model = init_detector(cfg, weight, device='cuda')
        all_detections = []
        all_annotations = []
        total_time = 0
        count = 0
        for idx, data in enumerate(dataset):
            # test a single image
            t = time.time()
            all_annotation = all_annotation_from_instance(data)
            all_detection = [[], []]

            result = inference_detector(model, data['filename'])
            result = [np.array([y for y in x if y[-1] > 0.3]) for x in result]
            for i in range(2):
                all_detection[i] = result[i]
            if idx > 10:
                total_time += time.time() - t
                count += 1
            all_annotations.append(all_annotation)
            all_detections.append(all_detection)
        res = {'name': name,
               'weight': weight,
               'config': config,
               # 'all_detections': [[ann.tolist() for ann in anns] for anns in all_detections],
               'average_precisions': {},
               'map': {},
               'weighted_map': {},
               'time': total_time / count,
               }
        print(res['time'])
        for thresh in [0.3, 0.5, 0.7, 0.9]:
            t = time.time()
            average_precisions, total_instances = evaluate(all_detections, all_annotations, 2, iou_threshold=thresh)
            res['average_precisions'][thresh] = average_precisions
            # weighted_map = sum([a * b for a, b in zip(total_instances, average_precisions)]) / sum(total_instances)
            # print(f'weighted map: {weighted_map:.4f}')
            total_precisions = [0, 0]
            weighed_precisions = 0
            print(thresh)
            for label, average_precision in average_precisions.items():
                total_precisions[label] = average_precision
                weighed_precisions += average_precision * total_instances[label] / sum(total_instances)
                print(['mif', 'ov'][label] + ': {:.4f}'.format(average_precision))
            mAP = sum(total_precisions) / 2
            print('mAP: {:.4f}'.format(mAP))  # mAP: 0.5000
            print('weighted mAP: {:.4f}'.format(weighed_precisions))  # mAP: 0.5000
            res['map'][thresh] = mAP
            res['weighted_map'][thresh] = weighed_precisions
        results[name] = res
        json.dump(results, open('evaluate.json', 'w'))
