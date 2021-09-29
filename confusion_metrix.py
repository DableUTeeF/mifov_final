from mmdet.apis import inference_detector, init_detector
from mmcv import Config
import json
import numpy as np


def compute_overlap(a, b):
    area = (b[2] - b[0]) * (b[3] - b[1])
    iw = np.minimum(np.expand_dims(a[2], axis=0), b[2]) - np.maximum(np.expand_dims(a[0], axis=0), b[0])
    ih = np.minimum(np.expand_dims(a[3], axis=0), b[3]) - np.maximum(np.expand_dims(a[1], axis=0), b[1])
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    ua = np.expand_dims((a[2] - a[0]) * (a[3] - a[1]), axis=0) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)
    intersection = iw * ih
    return intersection / ua


def confusion_metrix(results: list, ann: dict, confidence_threshold: float=0.3, iou_threshold:float=0.5):
    """
    :param results: result from `inference_detector()`
    :param ann: data['ann'] from the json
    :return: confusion metrix of mif, ov, bg
    """
    classes = ('mif', 'ov')
    predicts = {'mif': {'mif': 0, 'ov': 0, 'bg': 0},  # this is th gt
                'ov': {'mif': 0, 'ov': 0, 'bg': 0},   # for example predicts['mif'] is when mif is the gt
                'bg': {'mif': 0, 'ov': 0, 'bg': 0}
                }
    # loop over the predictions
    seens = {}  # the index of ann box seen
    for pred_class, r in enumerate(results):
        for result in r:
            pred_box = result[:-1]
            confidence = result[-1]
            found = False
            for ann_idx, ann_box in enumerate(ann['bboxes']):
                iou = compute_overlap(pred_box, ann_box)
                if iou > iou_threshold and confidence > confidence_threshold:
                    ann_class = ann['labels'][ann_idx]
                    if ann_idx not in seens:
                        seens[ann_idx] = (iou, pred_class)
                        predicts[classes[ann_class]][classes[pred_class]] += 1
                    elif iou == 1:
                        if seens[ann_idx][1] == ann_class:  # the answer is already correct
                            pass
                        else:
                            predicts[classes[ann_class]][classes[seens[ann_idx][1]]] -= 1
                            predicts[classes[ann_class]][classes[ann_class]] += 1
                            seens[ann_idx] = (iou, ann_class)
                    else:
                        if iou > seens[ann_idx][0]:
                            if seens[ann_idx][1] == pred_class:  # just the same box
                                pass
                            else:
                                predicts[classes[ann_class]][classes[seens[ann_idx][1]]] -= 1
                                seens[ann_idx] = (iou, pred_class)
                                predicts[classes[ann_class]][classes[pred_class]] += 1
                    found = True
                    break
            if not found and confidence > 0.5:
                predicts['bg'][classes[pred_class]] += 1

    # in case some gt don't get predicted
    for ann_idx, ann_box in enumerate(ann['bboxes']):
        if ann_idx not in seens:
            ann_class = ann['labels'][ann_idx]
            predicts[classes[ann_class]]['bg'] += 1
    return predicts


def cf_dict2np(cf):
    """
     p|p|p
    t
    t
    t
    """
    npcf = np.zeros((3, 3), dtype='uint32')
    for gt_class in cf:
        gt = cf[gt_class]
        gt_idx = classes[gt_class]
        for pd_class in gt:
            npcf[gt_idx, classes[pd_class]] += gt[pd_class]
    return npcf


if __name__ == '__main__':
    dataset = json.load(open('/home/palm/PycharmProjects/mmdetection/anns/test.json'))
    cfg = Config.fromfile('/home/palm/PycharmProjects/pig/mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py')
    # cfg.model.bbox_head.num_classes = 2
    model = init_detector(cfg, '/media/palm/BiggerData/algea/weights/cascade_101_lab_1/epoch_20.pth', device='cuda')
    cfs = np.zeros((3, 3), dtype='uint32')  # [gt, pd]
    classes = {'mif': 0, 'ov': 1, 'bg': 2}
    for i, data in enumerate(dataset):
        result = inference_detector(model, data['filename'])
        cf = confusion_metrix(result, data['ann'], confidence_threshold=0.3, iou_threshold=0.5)
        npcf = cf_dict2np(cf)
        cfs += npcf
        print(cfs)
