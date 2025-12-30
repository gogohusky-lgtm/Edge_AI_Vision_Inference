import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np

# 資料夾設定
IMAGE_DIR = 'images'
ANNOTATION_DIR = 'annotations/xmls'
OUTPUT_DIR = 'headshots'
CAT_DIR = os.path.join(OUTPUT_DIR, 'cats')
DOG_DIR = os.path.join(OUTPUT_DIR, 'dogs')

# 建立輸出資料夾
os.makedirs(CAT_DIR, exist_ok=True)
os.makedirs(DOG_DIR, exist_ok=True)

# 貓的品種清單（Oxford IIIT Pet Dataset 中的 12 種貓）
CAT_BREEDS = {
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll',
    'Russian_Blue', 'Siamese', 'Sphynx'
}

def parse_bounding_box(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find('object')
        if obj is None:
            return None
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        return xmin, ymin, xmax, ymax
    except Exception as e:
        print(f'[錯誤] 解析 bounding box 失敗：{xml_path}，原因：{e}')
        return None

def get_breed_name_from_filename(filename):
    # 從檔名推斷品種名稱（移除編號）
    breed = '_'.join(filename.replace('.xml', '').split('_')[:-1])
    return breed

def crop_square(image, bbox):
    h, w, _ = image.shape
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    box_size = max(xmax - xmin, ymax - ymin)
    half = box_size // 2

    left = cx - half
    right = cx + half
    top = cy - half
    bottom = cy + half

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - w)
    pad_bottom = max(0, bottom - h)

    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    cropped = image[top:bottom, left:right]
    cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cropped

def process_all():
    total = 0
    saved = 0
    skipped = 0

    for filename in os.listdir(ANNOTATION_DIR):
        if not filename.endswith('.xml'):
            continue
        total += 1
        xml_path = os.path.join(ANNOTATION_DIR, filename)
        image_name = filename.replace('.xml', '.jpg')
        image_path = os.path.join(IMAGE_DIR, image_name)

        if not os.path.exists(image_path):
            print(f'[略過] 找不到圖片：{image_path}')
            skipped += 1
            continue

        bbox = parse_bounding_box(xml_path)
        if bbox is None:
            print(f'[略過] 缺少 bounding box：{xml_path}')
            skipped += 1
            continue

        breed = get_breed_name_from_filename(filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f'[略過] 無法讀取圖片：{image_path}')
            skipped += 1
            continue

        headshot = crop_square(image, bbox)

        if breed in CAT_BREEDS:
            output_path = os.path.join(CAT_DIR, image_name)
        else:
            output_path = os.path.join(DOG_DIR, image_name)

        cv2.imwrite(output_path, headshot)
        saved += 1
        print(f'[儲存] {output_path}')

    print('\n📊 處理統計')
    print(f'總 XML 檔案：{total}')
    print(f'成功儲存：{saved}')
    print(f'略過檔案：{skipped}')

if __name__ == '__main__':
    process_all()
