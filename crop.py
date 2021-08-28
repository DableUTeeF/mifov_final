import os
import json
import cv2
import numpy as np

if __name__ == '__main__':
    src = '/media/palm/data/MicroAlgae/22_11_2020/{stage}/{t}/{cls}'
    out = '/media/palm/data/MicroAlgae/22_11_2020/classification'
    for stage in ['train', 'val', 'test']:
        out_stage = 'train' if stage != 'test' else 'val'
        for cls in ['mif', 'ov']:
            mif = 0
            r = src.format(stage=stage, cls=cls, t='jsn')
            for file in os.listdir(r):
                ann = json.load(open(os.path.join(r, file)))
                imname = file[:-5]
                image = cv2.imread(os.path.join(src.format(stage=stage, cls=cls, t='images'), imname+'.jpg'))
                if image is None:
                    print(imname)
                for idx, shape in enumerate(ann['shapes']):
                    (x1, y1), (x2, y2) = np.sort(np.array(shape['points']).astype('int32'), 0)
                    cropped_image = image[y1:y2, x1:x2]
                    h, w, _ = cropped_image.shape
                    empty_image = np.zeros((max(cropped_image.shape), max(cropped_image.shape), 3))
                    empty_image[:h, :w, :] = cropped_image
                    cv2.imwrite(os.path.join(out, out_stage, cls, f'{imname}_{idx:02d}.jpg'), empty_image)
