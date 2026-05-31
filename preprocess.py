import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='Image name (full relative path, e.g., `raw/image-name.jpg`).')
args = parser.parse_args()
# test args
if args.filename: image_path = args.filename
else: raise ValueError('Missing arguments! Specify the path to the image to be segmented.')

def crop_fxd(img, top, bottom, left, right):
    h,w = img.shape
    return img[top:h-bottom,left:w-right,]

def crop_pct(img, pct=.1):
    height, width = img.shape[:2]
    start_row, start_col = int(height * pct), int(width * pct)
    end_row, end_col = int(height * (1.-pct)), int(width * (1.-pct))
    return img[start_row:end_row, start_col:end_col]

def preprocess(imgname):
    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError
    clahe = cv2.createCLAHE(clipLimit=8., tileGridSize=(10,10))
    img = clahe.apply(img)
    img = crop_fxd(img, 0, 40, 0, 0,)
    path = imgname.split('/')[1]
    cv2.imwrite('raw/'+path, img)

preprocess(image_path)