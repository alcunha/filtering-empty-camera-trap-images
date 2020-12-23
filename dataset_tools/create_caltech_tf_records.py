# Copyright 2020 Fagner Cunha
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import hashlib
import json
import contextlib2

from absl import app
from absl import flags
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

FLAGS = flags.FLAGS

flags.DEFINE_string(
      'images_file', default=None,
      help=('Json file containing all images')
)

flags.DEFINE_string(
      'annotations_file', default=None,
      help=('Json file containing bbox annotations in COCO format')
)

flags.DEFINE_string(
      'splits_file', default=None,
      help=('Json file containing dataset spliting')
)

flags.DEFINE_string(
    'dataset_base_dir', default=None,
    help=('Path to images dataset base directory'))

flags.DEFINE_string(
    'output_dir', default=None,
    help=('Path where tfrecords will be saved on')
)

flags.DEFINE_enum(
    'label_type', default='empty', enum_values=['empty', 'species'],
    help=('Whether the model uses empty/nonempty (empty) or species labels'))

flags.DEFINE_list(
    'empty_classes_list', default=['empty'],
    help=('List of classes to be included as empty class for empty/nonempty'
          'label type')
)

flags.DEFINE_integer(
    'nonempty_category_id', default=1,
    help=('Numeric id for nonempty category when using agnostic labels')
)

flags.DEFINE_string(
    'label_map_path', default=None,
    help=('Path to label map proto')
)

flags.DEFINE_integer(
    'images_per_shard', default=600,
    help=('Number of images per shard')
)

flags.DEFINE_bool(
    'shufle_images', default=True,
    help=('Shufle images before to write to tfrecords')
)

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments')
  )

flags.mark_flag_as_required('images_file')
flags.mark_flag_as_required('annotations_file')
flags.mark_flag_as_required('splits_file')
flags.mark_flag_as_required('dataset_base_dir')
flags.mark_flag_as_required('label_map_path')
flags.mark_flag_as_required('output_dir')

def _get_image_dimensions_from_file(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  height = tf.shape(image)[0]
  width = tf.shape(image)[1]

  return height, width

def create_tf_example(image,
                      dataset_base_dir,
                      annotations,
                      category_index,
                      original_category_index,
                      images_resized=True):
  num_annotations_skipped = 0
  num_empty_annotations_skipped = 0
  filename = image['file_name'].split('/')[-1]
  image_id = image['id']

  image_path = os.path.join(dataset_base_dir, filename)
  if not tf.io.gfile.exists(image_path):
    return None, num_annotations_skipped, num_empty_annotations_skipped

  with tf.io.gfile.GFile(image_path, 'rb') as image_file:
    encoded_image_data = image_file.read()
  key = hashlib.sha256(encoded_image_data).hexdigest()

  orig_height = image['height']
  orig_width = image['width']
  if images_resized:
    height, width = _get_image_dimensions_from_file(image_path)
  else:
    height = image['height']
    width = image['width']

  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  classes_text = []
  classes = []
  for annotation in annotations:
    (x, y, bbox_width, bbox_height) = tuple(annotation['bbox'])
    category_id = annotation['category_id']

    if bbox_width <= 0 or bbox_height <= 0:
      num_annotations_skipped += 1
      continue

    if x < 0:
      bbox_width = bbox_width + x
      x = 0
    if y < 0:
      bbox_height = bbox_height + y
      y = 0
    if x + bbox_width > orig_width:
      bbox_width = orig_width - x
    if y + bbox_height > orig_height:
      bbox_height = orig_height - y

    if FLAGS.label_type=='empty':
      origin_category_text = original_category_index[category_id]['name']
      if origin_category_text in FLAGS.empty_classes_list:
        num_annotations_skipped += 1
        num_empty_annotations_skipped += 1
        continue

      category_id = int(FLAGS.nonempty_category_id)
      classes_text.append(category_index[category_id]['name'].encode('utf8'))
      classes.append(category_id)

    xmins.append(float(x) / orig_width)
    xmaxs.append(float(x + bbox_width) / orig_width)
    ymins.append(float(y) / orig_height)
    ymaxs.append(float(y + bbox_height) / orig_height)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))

  return tf_example, num_annotations_skipped, num_empty_annotations_skipped

def create_tf_record_from_images_list(images,
                                      annotations_index,
                                      dataset_base_dir,
                                      category_index,
                                      original_category_index,
                                      output_path):
  num_shards = 1 + (len(images) // FLAGS.images_per_shard)
  total_annot_skipped = 0
  total_empty_annot_skipped = 0
  total_image_skipped = 0

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)

    for index, image in enumerate(images):
      image_id = image['id']
      if image_id not in annotations_index:
        annotations_index[image_id] = []
      tf_example, annot_skipped, empty_annot_skipped = create_tf_example(
                                        image,
                                        dataset_base_dir,
                                        annotations_index[image_id],
                                        category_index,
                                        original_category_index)
      total_annot_skipped += annot_skipped
      total_empty_annot_skipped += empty_annot_skipped

      if tf_example is not None:
        output_shard_index = index % num_shards
        output_tfrecords[output_shard_index].write(
            tf_example.SerializeToString())
      else:
        total_image_skipped += 1

    tf.compat.v1.logging.info('Finished writing, skipped %d bboxes.',
                               total_annot_skipped)
    tf.compat.v1.logging.info('Skipped %d bboxes on empty images.', 
                              total_empty_annot_skipped)
    tf.compat.v1.logging.info('%d images not found.', total_image_skipped)

def _get_caltech_splits_by_location(caltech_splits_file):
  with tf.io.gfile.GFile(caltech_splits_file, 'r') as json_file:
    json_data = json.load(json_file)

  locations_map = { loc: split for split, locations in json_data.items()
                               for loc in locations}

  return locations_map

def _get_caltech_images_by_split(caltech_images_file, caltech_splits_file):
  with tf.io.gfile.GFile(caltech_images_file, 'r') as json_file:
    json_data = json.load(json_file)
  all_caltech_images = json_data['images']

  locations_map = _get_caltech_splits_by_location(caltech_splits_file)

  images_per_split = {}
  for image in all_caltech_images:
    split = locations_map[image['location']]
    if split not in images_per_split:
      images_per_split[split] = []
    images_per_split[split].append(image)

  return images_per_split

def _get_caltech_annotations_index(caltech_annotations_file):
  with tf.io.gfile.GFile(caltech_annotations_file, 'r') as json_file:
    json_data = json.load(json_file)
  annotations = json_data['annotations']
  original_category_index = label_map_util.create_category_index(
        json_data['categories'])

  annotations_index = {}
  for annotation in annotations:
    image_id = annotation['image_id']
    if image_id not in annotations_index:
      annotations_index[image_id] = []
    annotations_index[image_id].append(annotation)

  return annotations_index, original_category_index

def _create_caltech_tf_record_splits(caltech_images_file,
                                     caltech_annotations_file,
                                     caltech_splits_file,
                                     dataset_base_dir,
                                     label_map_path,
                                     output_path):

  images_per_split = _get_caltech_images_by_split(caltech_images_file,
                                                  caltech_splits_file)
  (annot_index, origin_cat_index) = _get_caltech_annotations_index(
              caltech_annotations_file)
  category_index = label_map_util.create_category_index_from_labelmap(
        label_map_path)

  for split, images in images_per_split.items():
    tf.compat.v1.logging.info('Started writing %s split.', split)
    tfrecord_path = os.path.join(output_path, 'caltech_%s.record' % split)

    if FLAGS.shufle_images:
      random.shuffle(images)

    create_tf_record_from_images_list(images,
                                      annot_index,
                                      dataset_base_dir,
                                      category_index,
                                      origin_cat_index,
                                      tfrecord_path)

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  if FLAGS.label_type=='species':
    raise RuntimeError('species option for flag --label_type not implemented')

  set_random_seeds()

  _create_caltech_tf_record_splits(FLAGS.images_file,
                                   FLAGS.annotations_file,
                                   FLAGS.splits_file,
                                   FLAGS.dataset_base_dir,
                                   FLAGS.label_map_path,
                                   FLAGS.output_dir)

if __name__ == '__main__':
  app.run(main)
