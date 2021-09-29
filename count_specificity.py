import os

total = {}
imgs = {}
root = '/media/palm/BiggerData/algea/predict_4'
for path in os.listdir(root):
    ims = 0
    for file in os.listdir(os.path.join(root, path)):
        if 'total_' in file:
            t = int(file.split('_')[1].split('.')[0])
            total[path] = t
        elif '_' in file and file[0] != '0':
            ims += 1
    imgs[path] = ims
print(imgs)
print(total)
