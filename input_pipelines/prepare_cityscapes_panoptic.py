import os
import json
import sys
from glob import glob
# TODO(): assert that panopticapi is in path
from panopticapi import semantic_data

PREPARE_IMAGE_FOLDERS = False
PREPARE_FILENAMES = False
PREPARE_INSTANCE = True
PREPARE_SEMANTIC = False
PREPARE_PROBLEM_DEF = False

# Move images from subdirs to main dir
def prepare_image_folders(dirs):
  for directory in dirs:
    paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    for path in paths:
      filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
      for filename in filenames:
        os.rename(os.path.join(path, filename), os.path.join(directory, filename))

# Convert panoptic annotations into semantic segmentation annotations
def prepare_semantic_data(in_dirs, out_dirs, in_files):
  for in_dir, out_dir, in_file in zip(in_dirs, out_dirs, in_files):
    semantic_data.extract_semantic(json_file=in_file,
                                   segmentations_folder=in_dir,
                                   semantic_seg_folder=out_dir)

#  Convert panoptic .json file to instance data .txt files for training & validation
# Include instance id, class id, bbox
def prepare_instance_data(json_files, out_dirs):
  thing_categories = []
  for json_file, out_dir in zip(json_files, out_dirs):
    with open(json_file, 'r') as f:
      data = json.load(f)
      annotations = data['annotations']
      categories = data['categories']
      for category in categories:
        if category['isthing'] == 1:
          thing_categories.append(category['id'])
      print(thing_categories)
      for annotation in annotations:
        filename = os.path.splitext(annotation['file_name'])[0]
        with open(os.path.join(out_dir, filename + '.txt'), 'w') as txt_file:
          for segment_info in annotation['segments_info']:
            c_id = segment_info['category_id']
            if c_id in thing_categories:
              # if int(segment_info['iscrowd']) == 0:
              #   i_id = segment_info['id']
              #   bbox = segment_info['bbox']
              #   bbox_str = str(bbox[0]) + " " + str(bbox[1]) + " "  + str(bbox[2]) + " "  + str(bbox[3])
              #   txt_file.write(str(i_id) + " " + str(c_id) + " " + bbox_str + "\n")
              # else:
              #   print(segment_info['id'])
              i_id = segment_info['id']
              bbox = segment_info['bbox']
              bbox_str = str(bbox[0]) + " " + str(bbox[1]) + " "  + str(bbox[2]) + " "  + str(bbox[3])
              if int(segment_info['iscrowd']) == 0:
                weight = 1
              else:
                weight = 0
              txt_file.write(str(i_id) + " " + str(c_id) + " " + bbox_str + " " + str(weight) + "\n")


# Put filename list in .lst or .txt file for training, validation & testing
def prepare_filename_list(directories, save_dirs):
  for directory, save_dir in zip(directories, save_dirs):
    filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    save_file = os.path.join(save_dir, 'filenames.lst')
    with open(save_file, 'w') as f:
      for filename in filenames:
        f.write(os.path.splitext(filename)[0] + '\n')

# Print information for problem definition
def prepare_problem_def(json_file, ignore_ids, void_info):
  id_count = 0
  object_id_count = 0

  ids_total = list()
  names_total = list()
  colors_total = list()
  class_ids_total = list()
  is_objects_total = list()

  object_ids_total = list()
  object_cids_total = list()
  object_cids_real_total = list()

  lids2cids = list()

  cids2labels = list()
  cids2colors = list()
  cids2lids = list()
  cids2is_object = list()
  lids2object_cids = list()
  object_cids2colors = list()
  object_cids2labels = list()
  object_cids2lids = list()
  with open(json_file, 'r') as f:
    categories = json.load(f)
  for category in categories:
    ids_total.append(category['id'])
    names_total.append(category['name'])
    colors_total.append(category['color'])
    if category['id'] not in ignore_ids:
      class_ids_total.append(id_count)
    else:
      class_ids_total.append(-1)
    is_objects_total.append(category['isthing'])
    if category['isthing'] == 1:
      if category['id'] not in ignore_ids:
        object_ids_total.append(category['id'])
        object_cids_total.append(id_count)
        object_cids_real_total.append(object_id_count)
        object_id_count += 1
    if category['id'] not in ignore_ids:
      id_count += 1

  for i in range(max(ids_total) + 1):
    if i in ids_total:
      index = ids_total.index(i)
      lids2cids.append(class_ids_total[index])
    else:
      lids2cids.append(-1)

  for i in range(max(ids_total) + 1):
    if i in object_ids_total:
      index = object_ids_total.index(i)
      lids2object_cids.append(object_cids_real_total[index])
    else:
      lids2object_cids.append(-1)

  print('lids2cids:')
  print(str(lids2cids) + '\n')

  for x in range(max(lids2cids) + 1):
    try:
      index = class_ids_total.index(x)
      [r, g, b] = colors_total[index]
      cids2colors.append([r, g, b])
      cids2labels.append(names_total[index])
      cids2lids.append(ids_total[index])
      cids2is_object.append(is_objects_total[index])
    except ValueError:
      print('Not all class labels from 0 up to ' + str(max(lids2cids)) + ' exist.')
      print('cids2colors, cids2labels and cids2lids could not be computed! \n')

  for x in range(max(lids2object_cids) + 1):
    try:
      index = class_ids_total.index(object_cids_total[x])
      [r, g, b] = colors_total[index]
      object_cids2colors.append([r, g, b])
      object_cids2labels.append(names_total[index])
      object_cids2lids.append(ids_total[index])
    except ValueError:
      print('Not all class labels from 0 up to ' + str(max(lids2cids)) + ' exist.')
      print('cids2colors, cids2labels and cids2lids could not be computed! \n')


  if void_info[0]:
    cids2lids.append(-1)
    cids2is_object.append(0)
    cids2labels.append(void_info[1])
    cids2colors.append(void_info[2])


  print('cids2labels:')
  print(str(cids2labels) + '\n')

  print('cids2colors:')
  print(str(cids2colors) + '\n')

  print('cids2lids:')
  print(str(cids2lids) + '\n')

  print('cids2is_object:')
  print(str(cids2is_object) + '\n')

  print('lids2object_cids:')
  print(str(lids2object_cids) + '\n')

  print('object_cids2colors:')
  print(str(object_cids2colors) + '\n')

  print('object_cids2labels:')
  print(str(object_cids2labels) + '\n')

  print('object_cids2lids:')
  print(str(object_cids2lids) + '\n')

  print(categories)

if __name__ == "__main__":
  data_dir = '/home/ddegeus/datasets/Cityscapes/'

  image_folder = 'images'
  panoptic_folder = 'panoptic'
  panoptic_json_path = 'panoptic.json'

  categories_json_path = 'cityscapes_categories.json'

  semantic_proc_dir = 'panoptic_proc'
  # instance_proc_dir = 'panoptic_txt'
  # instance_proc_dir = 'panoptic_txt_no_crowd'
  instance_proc_dir = 'panoptic_txt_weights'
  filename_list_dir = 'panoptic'

  fn_filename = 'filenames.lst'

  train_dir = 'training'
  val_dir = 'validation'
  test_dir = 'testing'

  ignore_category_ids = []
  add_void_category = True
  void_color = [0, 0, 0]
  void_label = "Unlabeled"

  void_info = [add_void_category, void_label, void_color]

  if PREPARE_IMAGE_FOLDERS:
    splits = [train_dir, val_dir, test_dir]

    dirs = list()
    for split in splits:
      dirs.append(os.path.join(data_dir, os.path.join(split, image_folder)))

    prepare_image_folders(dirs)

  if PREPARE_SEMANTIC:
    in_dirs = list()
    out_dirs = list()
    in_files = list()
    splits = [train_dir, val_dir]

    for split in splits:
      in_dirs.append(os.path.join(data_dir, os.path.join(split, panoptic_folder)))
      out_dirs.append(os.path.join(data_dir, os.path.join(split, semantic_proc_dir)))
      in_files.append(os.path.join(data_dir, os.path.join(split, panoptic_json_path)))

    prepare_semantic_data(in_dirs, out_dirs, in_files)

  if PREPARE_FILENAMES:
    directories = list()
    save_dirs = list()
    splits = [train_dir, val_dir, test_dir]

    for split in splits:
      directories.append(os.path.join(data_dir, os.path.join(split, image_folder)))
      save_dirs.append(os.path.join(data_dir, os.path.join(split, filename_list_dir)))

    prepare_filename_list(directories, save_dirs)

  if PREPARE_INSTANCE:
    json_files = list()
    out_dirs = list()

    splits = [train_dir, val_dir]

    for split in splits:
      json_files.append(os.path.join(data_dir, os.path.join(split, panoptic_json_path)))
      out_dirs.append(os.path.join(data_dir, os.path.join(split, instance_proc_dir)))

    prepare_instance_data(json_files, out_dirs)

  if PREPARE_PROBLEM_DEF:
    json_file = os.path.join(data_dir, categories_json_path)
    prepare_problem_def(json_file, ignore_category_ids, void_info)
