import os
import cv2

# 原始與輸出資料夾設定
SOURCE_DIR = 'headshots'
TARGET_DIR = 'resized'
CATEGORIES = ['cats', 'dogs']
TARGET_SIZE = (160, 160)

# 建立輸出資料夾
for category in CATEGORIES:
    os.makedirs(os.path.join(TARGET_DIR, category), exist_ok=True)

def resize_images():
    skipped = 0
    resized = 0

    for category in CATEGORIES:
        src_path = os.path.join(SOURCE_DIR, category)
        dst_path = os.path.join(TARGET_DIR, category)

        for filename in os.listdir(src_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image_path = os.path.join(src_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f'[略過] 無法讀取圖片：{image_path}')
                skipped += 1
                continue

            h, w = image.shape[:2]
            if h < TARGET_SIZE[1] or w < TARGET_SIZE[0]:
                print(f'[略過] 圖片太小：{filename}（{w}x{h}）')
                skipped += 1
                continue

            resized_img = cv2.resize(image, TARGET_SIZE)
            output_path = os.path.join(dst_path, filename)
            cv2.imwrite(output_path, resized_img)
            resized += 1
            print(f'[儲存] {output_path}')

    print('\n📊 縮放統計')
    print(f'成功縮放：{resized}')
    print(f'略過圖片：{skipped}')

if __name__ == '__main__':
    resize_images()
