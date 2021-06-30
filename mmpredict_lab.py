import sys
# sys.path.extend(['/home/palm/PycharmProjects/mmdetection'])
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
import cv2
import json
import os


if __name__ == '__main__':
    dataset = json.load(open('/home/palm/PycharmProjects/mmdetection/anns/test.json'))

    cfg = Config.fromfile('/home/palm/PycharmProjects/pig/mmdetection/configs/gfl/gfl_r50_fpn_mstrain_2x_coco.py')
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
    model = init_detector(cfg, '/media/palm/BiggerData/algea/weights/gfl_r50/epoch_24.pth', device='cuda')
    os.makedirs('/media/palm/BiggerData/algea/gfl_r50_lab_1', exist_ok=True)
    for data in dataset:
        # test a single image
        result = inference_detector(model, data['filename'])
        # show the results
        img = model.show_result(data['filename'],
                                result,
                                score_thr=0.3, show=False)
        # print(result)
        cv2.imwrite(os.path.join('/media/palm/BiggerData/algea/gfl_r50_lab_1',
                                 os.path.basename(data['filename'])),
                    img)
        # cv2.imshow(data['filename'], cv2.resize(img, None, None, 0.5, 0.5))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # show_result_pyplot(model, data['filename'], result, score_thr=0.3)
