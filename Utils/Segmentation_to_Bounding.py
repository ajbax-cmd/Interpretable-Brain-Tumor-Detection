import cv2
import numpy as np
import os
import json

#Script to convert segmentation annotations to YOLO bounding box annotations - used for Br35h
def points_to_mask(points, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    polygon = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [polygon], 255)
    return mask

def ellipse_to_mask(cx, cy, rx, ry, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    center = (int(cx), int(cy))
    axes = (int(rx), int(ry))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask

def circle_to_mask(cx, cy, r, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    center = (int(cx), int(cy))
    radius = int(r)
    cv2.circle(mask, center, radius, 255, -1)
    return mask

def mask_to_yolo_bbox(mask, image_shape):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contours found.")
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    img_height, img_width = image_shape
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height

    return x_center, y_center, width, height

def read_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def process_masks_from_json(json_path, output_dir, image_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = read_json_file(json_path)
    
    for image_id, content in data.items():
        image_filename = content['filename']
        image_path = os.path.join(image_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Image {image_filename} not found in {image_dir}.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_filename}.")
            continue
        
        height, width = image.shape[:2]

        for region in content['regions']:
            shape_attributes = region['shape_attributes']
            mask = None
            if shape_attributes['name'] == 'polygon':
                points = list(zip(shape_attributes['all_points_x'], shape_attributes['all_points_y']))
                points = [(int(x), int(y)) for x, y in points]
                mask = points_to_mask(points, (height, width))
            elif shape_attributes['name'] == 'ellipse':
                cx = shape_attributes['cx']
                cy = shape_attributes['cy']
                rx = shape_attributes['rx']
                ry = shape_attributes['ry']
                mask = ellipse_to_mask(cx, cy, rx, ry, (height, width))
            elif shape_attributes['name'] == 'circle':
                cx = shape_attributes['cx']
                cy = shape_attributes['cy']
                r = shape_attributes['r']
                mask = circle_to_mask(cx, cy, r, (height, width))

            if mask is not None:
                bbox = mask_to_yolo_bbox(mask, (height, width))
                if bbox:
                    yolo_filename = os.path.splitext(image_filename)[0] + ".txt"
                    yolo_filepath = os.path.join(output_dir, yolo_filename)
                    
                    with open(yolo_filepath, 'w') as f:
                        class_id = 0
                        f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                    print(f"Saved annotation to: {yolo_filepath}")
                else:
                    print(f"No bounding box found for image: {image_filename}")
            else:
                print(f"Unsupported shape: {shape_attributes['name']}")


json_path = "/home/alan/Documents/YOLOv9/yolov9/Dataset_1/brain-tumor-detection-dataset/Br35H-Mask-RCNN/validate/annotations_validate.json"
output_dir = "/home/alan/Documents/YOLOv9/yolov9/Dataset_1/brain-tumor-detection-dataset/Br35H-Mask-RCNN/validate/labels"
image_dir = "/home/alan/Documents/YOLOv9/yolov9/Dataset_1/brain-tumor-detection-dataset/Br35H-Mask-RCNN/validate/images"
process_masks_from_json(json_path, output_dir, image_dir)
