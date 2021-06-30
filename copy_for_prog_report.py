import os
import shutil


if __name__ == '__main__':
    out = '/media/palm/Works/tgist_finals/images'
    wanted = os.listdir(out)
    src = '/media/palm/BiggerData/algea/predict_2'

    for model in os.listdir(src):
        for file in wanted:
            shutil.copy(os.path.join(src, model, 'kk', file + '.jpg'),
                        os.path.join(out, file, model + '.jpg'))
