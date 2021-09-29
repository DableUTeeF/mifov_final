from mmdet.datasets.pipelines.transforms import Resize
import cv2

if __name__ == '__main__':
    transform = Resize([(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                       multiscale_mode='value')
    image = cv2.imread('/media/palm/data/MicroAlgae/22_11_2020/images/00005.jpg')
    cv2.imwrite('/media/palm/BiggerData/algea/transforms/resize/ori.png', image)
    for i in range(100):
        results = {'img': image.copy()}
        transform._random_scale(results)
        transform._resize_img(results)
        cv2.imwrite(f"/media/palm/BiggerData/algea/transforms/resize/{i:03d}.png", results['img'])
