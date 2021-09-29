import sys
# sys.path.extend(['/home/palm/PycharmProjects/mmdetection'])
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
import cv2
import json
import os


if __name__ == '__main__':
    filedir = '/media/palm/data/MicroAlgae/specificity'
    out_path = '/media/palm/BiggerData/algea/predict_4/gfl_r50_kk_1'

    cfg = Config.fromfile('/home/palm/PycharmProjects/pig/mmdetection/configs/detr/detr_r50_8x2_150e_coco.py')
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
    model = init_detector(cfg, '/media/palm/BiggerData/algea/weights/detr_lab_4/epoch_30.pth', device='cuda')
    os.makedirs(out_path, exist_ok=True)
    for file in os.listdir(filedir):
        filename = os.path.join(filedir, file)

        # test a single image
        result = inference_detector(model, filename)
        # show the results
        img = model.show_result(filename,
                                result,
                                score_thr=0.3, show=False)
        # print(result)
        cv2.imwrite(os.path.join(out_path,
                                 file),
                    img)
        # cv2.imshow(data['filename'], cv2.resize(img, None, None, 0.5, 0.5))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # show_result_pyplot(model, data['filename'], result, score_thr=0.3)
