import torch
import numpy as np

import heapq

import matplotlib.pyplot as plt
import os
import copy
import time

import cv2

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    #plt.imshow(img_rgb_float)
    #plt.axis('off')
    plt.imsave(fname, img_rgb_float)

class ImagePatch:
    def __init__(self, patch, label, distance, image_id, original_img=None, act_pattern=None, patch_indices=None):
        self.patch = patch
        self.label = label
        self.negative_distance = -distance
        self.image_id = image_id  # Added this field to store original image id that patch comes from

        self.original_img = original_img
        self.act_pattern = act_pattern
        self.patch_indices = patch_indices

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance

class ImagePatchInfo:
    def __init__(self, label, distance, image_id):
        self.label = label
        self.negative_distance = -distance
        self.image_id = image_id   # Added this field to store original image id that patch comes from

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


# find the nearest patches in the dataset to each prototype
def find_k_nearest_patches_to_prototypes(dataloader, prototype_network_parallel, k=5, preprocess_input_function=None, full_save=False, root_dir_for_saving_images='./nearest', log=print, prototype_activation_function_in_numpy=None):
    prototype_network_parallel.eval()
    log('find nearest patches')
    start = time.time()
    n_prototypes = prototype_network_parallel.module.num_prototypes
    prototype_shape = prototype_network_parallel.module.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info

    heaps = [ [] for _ in range(n_prototypes) ]

    for idx, (search_batch_input, search_y, image_paths) in enumerate(dataloader):
        print(f'batch {idx}')
        search_batch = preprocess_input_function(search_batch_input) if preprocess_input_function is not None else search_batch_input
        with torch.no_grad():
            search_batch = search_batch.cuda()
            protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)
        proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

        for img_idx, distance_map in enumerate(proto_dist_):
            for j in range(n_prototypes):
                closest_patch_distance_to_prototype_j = np.amin(distance_map[j])
                image_id = image_paths[img_idx]  # Store image path

                if full_save:
                    closest_patch_indices_in_distance_map_j = list(np.unravel_index(np.argmin(distance_map[j], axis=None), distance_map[j].shape))
                    closest_patch_indices_in_distance_map_j = [0] + closest_patch_indices_in_distance_map_j
                    closest_patch_indices_in_img = compute_rf_prototype(search_batch.size(2), closest_patch_indices_in_distance_map_j, protoL_rf_info)
                    closest_patch = search_batch_input[img_idx, :, closest_patch_indices_in_img[1]:closest_patch_indices_in_img[2], closest_patch_indices_in_img[3]:closest_patch_indices_in_img[4]]
                    closest_patch = closest_patch.numpy()
                    closest_patch = np.transpose(closest_patch, (1, 2, 0))
                    original_img = search_batch_input[img_idx].numpy()
                    original_img = np.transpose(original_img, (1, 2, 0))

                    if prototype_network_parallel.module.prototype_activation_function == 'log':
                        act_pattern = np.log((distance_map[j] + 1)/(distance_map[j] + prototype_network_parallel.module.epsilon))
                    elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                        act_pattern = max_dist - distance_map[j]
                    else:
                        act_pattern = prototype_activation_function_in_numpy(distance_map[j])

                    patch_indices = closest_patch_indices_in_img[1:5]

                    closest_patch = ImagePatch(patch=closest_patch, label=search_y[img_idx], distance=closest_patch_distance_to_prototype_j, image_id=image_id, original_img=original_img, act_pattern=act_pattern, patch_indices=patch_indices)
                else:
                    closest_patch = ImagePatchInfo(label=search_y[img_idx], distance=closest_patch_distance_to_prototype_j, image_id=image_id)

                if len(heaps[j]) < k:
                    heapq.heappush(heaps[j], closest_patch)
                else:
                    heapq.heappushpop(heaps[j], closest_patch)

    for j in range(n_prototypes):
        heaps[j].sort()
        heaps[j] = heaps[j][::-1]

        if full_save:
            dir_for_saving_images = os.path.join(root_dir_for_saving_images, str(j))
            makedir(dir_for_saving_images)
            labels = []

            for i, patch in enumerate(heaps[j]):
                np.save(os.path.join(dir_for_saving_images, 'nearest-' + str(i+1) + '_act.npy'), patch.act_pattern)
                plt.imsave(fname=os.path.join(dir_for_saving_images, 'nearest-' + str(i+1) + '_original.png'), arr=patch.original_img, vmin=0.0, vmax=1.0)
                img_size = patch.original_img.shape[0]
                upsampled_act_pattern = cv2.resize(patch.act_pattern, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
                rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
                rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
                heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_pattern), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[...,::-1]
                overlayed_original_img = 0.5 * patch.original_img + 0.3 * heatmap
                plt.imsave(fname=os.path.join(dir_for_saving_images, 'nearest-' + str(i+1) + '_original_with_heatmap.png'), arr=overlayed_original_img, vmin=0.0, vmax=1.0)

                if patch.patch.shape[0] != img_size or patch.patch.shape[1] != img_size:
                    np.save(os.path.join(dir_for_saving_images, 'nearest-' + str(i+1) + '_receptive_field_indices.npy'), patch.patch_indices)
                    plt.imsave(fname=os.path.join(dir_for_saving_images, 'nearest-' + str(i+1) + '_receptive_field.png'), arr=patch.patch, vmin=0.0, vmax=1.0)
                    overlayed_patch = overlayed_original_img[patch.patch_indices[0]:patch.patch_indices[1], patch.patch_indices[2]:patch.patch_indices[3], :]
                    plt.imsave(fname=os.path.join(dir_for_saving_images, 'nearest-' + str(i+1) + '_receptive_field_with_heatmap.png'), arr=overlayed_patch, vmin=0.0, vmax=1.0)

                high_act_patch_indices = find_high_activation_crop(upsampled_act_pattern)
                high_act_patch = patch.original_img[high_act_patch_indices[0]:high_act_patch_indices[1], high_act_patch_indices[2]:high_act_patch_indices[3], :]
                np.save(os.path.join(dir_for_saving_images, 'nearest-' + str(i+1) + '_high_act_patch_indices.npy'), high_act_patch_indices)
                plt.imsave(fname=os.path.join(dir_for_saving_images, 'nearest-' + str(i+1) + '_high_act_patch.png'), arr=high_act_patch, vmin=0.0, vmax=1.0)
                imsave_with_bbox(fname=os.path.join(dir_for_saving_images, 'nearest-' + str(i+1) + '_high_act_patch_in_original_img.png'), img_rgb=patch.original_img, bbox_height_start=high_act_patch_indices[0], bbox_height_end=high_act_patch_indices[1], bbox_width_start=high_act_patch_indices[2], bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

            labels = np.array([patch.label for patch in heaps[j]])
            np.save(os.path.join(dir_for_saving_images, 'class_id.npy'), labels)

    labels_all_prototype = np.array([[patch.label for patch in heaps[j]] for j in range(n_prototypes)])
    if full_save:
        np.save(os.path.join(root_dir_for_saving_images, 'full_class_id.npy'), labels_all_prototype)

    end = time.time()
    log(f'\tfind nearest patches time: {end - start}')

    return heaps

