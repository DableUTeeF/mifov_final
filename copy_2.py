import os
# import shutil
import cv2

if __name__ == '__main__':
    out = '/media/palm/Works/thesis/dcn'
    wanted = os.listdir(out)
    src = '/media/palm/Works/thesis/field_data'

    for name in os.listdir(src):
        image = cv2.imread(os.path.join(src, name, 'vfnet_r50.jpg'))
        cv2.imwrite(os.path.join(out, name + '_base.jpg'), image)

        image = cv2.imread(os.path.join(src, name, 'vfnet_r50_dcn.jpg'))
        cv2.imwrite(os.path.join(out, name + '_dcn.jpg'), image)
