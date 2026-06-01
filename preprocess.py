import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='Image name (full relative path, e.g., `raw/image-name.jpg`).')
parser.add_argument('--barlength', type=float, help='Bar length.')
args = parser.parse_args()
# test args
if args.filename: image_path = args.filename
else: raise ValueError('Missing arguments! Specify the path to the image to be segmented.')

def crop_fxd(img, top=0, bottom=70, left=0, right=0):
    h,w = img.shape
    return img[top:h-bottom,left:w-right,]

def crop_pct(img, pct=.1):
    height, width = img.shape[:2]
    start_row, start_col = int(height * pct), int(width * pct)
    end_row, end_col = int(height * (1.-pct)), int(width * (1.-pct))
    return img[start_row:end_row, start_col:end_col]

def preprocess(img, contrast = False):
    img_new = crop_fxd(img,)
    if contrast:
        clahe = cv2.createCLAHE(clipLimit=8., tileGridSize=(12,12))
        img_new = clahe.apply(img_new)
    return img_new


def get_bar(img):
    h,w = img.shape
    bar = crop_fxd(img, top=h-70, bottom=40, left=1097, right=68)
    return bar

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None: raise ValueError

if args.barlength:
    length = args.barlength
    bar = get_bar(img)
    pixels = bar.shape[1]
    print(f'Ratio = {length/pixels:.4f}')

img = preprocess(img)

path = image_path.split('/')[1]
cv2.imwrite('raw/'+path, img)