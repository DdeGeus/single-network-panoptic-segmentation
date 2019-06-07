import numpy as np
import skimage.transform

def reshape_normalized_box(box_normalized,
                           image_size):
  x_min, y_min, x_max, y_max = box_normalized
  new_height, new_width = image_size
  x_min = x_min * new_width
  y_min = y_min * new_height
  x_max = x_max * new_width
  y_max = y_max * new_height

  return [x_min, y_min, x_max, y_max]

def resize(mask,
           size):
  return skimage.transform.resize(mask, size,
                           order=1, mode='constant', cval=0,
                           clip=True, preserve_range=False)

def reshape_single_mask(box_normalized,
                        mask,
                        class_id,
                        image_size,
                        threshold):

  [x_min, y_min, x_max, y_max] = reshape_normalized_box(box_normalized,
                                                      image_size)
  mask = mask[class_id]
  mask = resize(mask, (int(y_max) - int(y_min), int(x_max) - int(x_min)))
  mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

  full_mask = np.zeros(image_size, dtype=np.uint8)
  full_mask[int(y_min):int(y_max), int(x_min):int(x_max)] = mask

  return full_mask

def reshape_masks_to_image_size(boxes_normalized,
                                num_boxes,
                                classes,
                                masks,
                                image_size,
                                threshold=0.5):
  """

  Args:
    boxes_normalized: [N, 4]
    num_boxes: []
    masks: [N, Hm, Wm] (Usually [N, 33, 33])
    image_size: [2] = (H, W)
    threshold: []

  Returns: [N, H, W]

  """

  full_masks = list()
  for i in range(num_boxes):
    full_mask = reshape_single_mask(boxes_normalized[i],
                                    masks[i],
                                    classes[i],
                                    image_size,
                                    threshold)
    full_masks.append(full_mask)
  if num_boxes > 0:
    full_masks = np.stack(full_masks, axis=0)
  else:
    full_masks = np.zeros(image_size)

  return full_masks