import imgaug.augmenters as iaa
from utils.transforms import ImgAug

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            # iaa.Dropout([0.0, 0.01]),
            # iaa.Sharpen((0.0, 0.2)),

            iaa.Affine(rotate=(-45, 45), scale=(0.8,1.3), translate_percent=(-0.2,0.2)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.AddToBrightness((-100, 100)), 
            iaa.AddToHue((-128, 128)),
            iaa.Fliplr(0.5),
            iaa.AdditiveGaussianNoise(scale=(0,8)),
            iaa.MotionBlur(k=(3,10), angle=(-90,90)),
            iaa.Crop(percent=(0.0,0.3))
        ], random_order=False)