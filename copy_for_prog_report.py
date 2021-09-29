import os
# import shutil
import cv2

if __name__ == '__main__':
    out = '/media/palm/Works/thesis/field_data'
    wanted = os.listdir(out)
    src = '/media/palm/BiggerData/algea/predict_2'

    sizes = {
        'line_oa_chat_200909_105503': (int(210 // 0.4), int(360 // 0.4), int(480 // 0.4), int(480 // 0.4)),  # x1, y1, x2, y2
        'line_oa_chat_200909_105900': (int(60 // 0.4), int(210 // 0.4), int(240 // 0.4), int(280 // 0.4)),  # x1, y1, x2, y2
        'line_oa_chat_200909_110057': (int(165 // 0.4), int(225 // 0.4), int(275 // 0.4), int(290 // 0.4)),  # x1, y1, x2, y2
        'line_oa_chat_200909_110607': (int(160 // 0.4), int(180 // 0.4), int(265 // 0.4), int(245 // 0.4)),  # x1, y1, x2, y2
        '24_1': (int(86 // 0.4), int(388 // 0.4), int(250 // 0.4), int(485 // 0.4)),  # x1, y1, x2, y2
        '1_1': (50, 0, 550, 300),  # x1, y1, x2, y2
    }

    for model in os.listdir(src):
        for file in wanted:
            if 'old' in file:
                continue
            image = cv2.imread(os.path.join(src, model, 'kk', file + '.jpg'))
            if file in sizes:
                x1, y1, x2, y2 = sizes[file]
                image = image[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(out, file, model + '.jpg'), image)
            # shutil.copy(os.path.join(src, model, 'kk', file + '.jpg'),
            #             os.path.join(out, file, model + '.jpg'))
