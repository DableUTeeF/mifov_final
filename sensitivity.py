from mmdet.apis import inference_detector, init_detector
from mmcv import Config
import cv2
import json
import os
from confusion_metrix import confusion_metrix

if __name__ == '__main__':
    dataset = json.load(open('/home/palm/PycharmProjects/mmdetection/anns/test.json'))
    filedir = '/media/palm/data/MicroAlgae/specificity'
    path = (
        '/home/palm/PycharmProjects/pig/mmdetection/configs/',
        '/media/palm/BiggerData/algea/weights/',
        '/media/palm/BiggerData/algea/predict_4',
    )
    configs = [
        ('cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py',
         'cascade_101_lab_1/epoch_20.pth',
         'cascade_rcnn_r101'),
        ('cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py',
         'cascade_r50/epoch_20.pth',
         'cascade_rcnn_r50'),
        ('detr/detr_r50_8x2_150e_coco.py',
         'detr_lab_4/epoch_30.pth',
         'detr_r50'),
        ('retinanet/retinanet_r50_fpn_1x_coco.py',
         'retinanet_lab_1/epoch_23.pth',
         'retinanet_r50'),
        ('retinanet/retinanet_r101_fpn_1x_coco.py',
         'retinanet_r101/epoch_24.pth',
         'retinanet_r101'),
        ('faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py',
         'rcnn_lab_1/epoch_30.pth',
         'rcnn_r101'),
        ('faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco.py',
         'rcnn_r50/epoch_24.pth',
         'rcnn_r50'),
        ('yolo/yolov3_d53_mstrain-608_273e_coco.py',
         'yolov3_lab_1/epoch_30.pth',
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
    output = {}
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

        model = init_detector(cfg, weight, device='cuda')

        # JSON
        out_path = os.path.join(path[2], name, 'lab')
        os.makedirs(out_path, exist_ok=True)

        sensitve = {'mif': 0, 'ov': 0}
        total = {'mif': 0, 'ov': 0}
        false_alarm = {'mif': 0, 'ov': 0}

        for idx, data in enumerate(dataset):
            result = inference_detector(model, data['filename'])
            cf = confusion_metrix(result, data['ann'], confidence_threshold=0.3, iou_threshold=0.5)
            if data['ann']['labels'][0] == 0:
                total['mif'] += 1
            else:
                total['ov'] += 1

            if cf['mif']['mif'] > 0:
                sensitve['mif'] += 1
            elif cf['ov']['ov'] > 0:
                sensitve['ov'] += 1

            if cf['mif']['ov'] > 0 and cf['mif']['mif'] == 0:
                false_alarm['ov'] += 1
            elif cf['ov']['mif'] > 0 and cf['ov']['ov'] == 0:
                false_alarm['mif'] += 1

        for idx, file in enumerate(os.listdir(filedir)):
            filename = os.path.join(filedir, file)
            image = cv2.imread(filename)

            # test a single image
            result = inference_detector(model, image)
            mif_result = result[0][result[0][:, -1] > 0.3]
            ov_result = result[1][result[1][:, -1] > 0.3]
            if len(mif_result) > 0:
                false_alarm['mif'] += 1
            elif len(ov_result) > 0:
                false_alarm['ov'] += 1

        output[name] = {'sensitive': sensitve, 'false_alarm': false_alarm, 'total': total}
    json.dump(output, open('sens_spec.json', 'w'))
