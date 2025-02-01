import os
import tensorflow as tf
from pathlib import Path
import json
import matplotlib
import collections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from official.vision.data import tfrecord_lib
from official.vision.data.create_coco_tf_record import coco_annotations_to_lists
import tqdm

def get_category_map(annotation_path, num_classes):
  with Path(annotation_path).open() as f:
      data = json.load(f)

  category_map = {id+1: {'id': cat_dict['id'],
                       'name': cat_dict['name']}
                  for id, cat_dict in enumerate(data['categories'][:num_classes])}
  return category_map


class LvisAnnotation:

  def __init__(self, annotation_path):
    with Path(annotation_path).open() as f:
      data = json.load(f)
    self._data = data

    img_id2annotations = collections.defaultdict(list)
    for a in self._data.get('annotations', []):
      img_id2annotations[a['image_id']].append(a)
    self._img_id2annotations = {
        k: list(sorted(v, key=lambda a: a['id']))
        for k, v in img_id2annotations.items()
    }

  @property
  def categories(self):
    """Return the category dicts, as sorted in the file."""
    return self._data['categories']

  @property
  def images(self):
    """Return the image dicts, as sorted in the file."""
    sub_images = []
    for image_info in self._data['images']:
      if image_info['id'] in self._img_id2annotations:
        sub_images.append(image_info)
    return sub_images

  def get_annotations(self, img_id):
    """Return all annotations associated with the image id string."""
    # Some images don't have any annotations. Return empty list instead.
    return self._img_id2annotations.get(img_id, [])

def generate_tf_records(tf_record_path, prefix, images_path, annotation_file, num_shards=5):
    """Generate TFRecords."""

    lvis_annotation = LvisAnnotation(annotation_file)

    def _process_example(images_path, image_info, id_to_name_map):
      # Search image dirs.
      filename = image_info['filename']
      image = tf.io.read_file(os.path.join(images_path, filename))
      instances = lvis_annotation.get_annotations(img_id=image_info['id'])
      # print([x['category_id'] for x in instances])
      is_crowd = {'iscrowd': 0}
      instances = [dict(x, **is_crowd) for x in instances]
      neg_category_ids = image_info.get('neg_category_ids', [])
      not_exhaustive_category_ids = image_info.get(
          'not_exhaustive_category_ids', []
      )

      data, _ = coco_annotations_to_lists(instances,
                                          id_to_name_map,
                                          image_info['height'],
                                          image_info['width'],
                                          include_masks=True)
      # data['category_id'] = [id-1 for id in data['category_id']]

      keys_to_features = {
          'image/encoded':
              tfrecord_lib.convert_to_feature(image.numpy()),
          'image/filename':
               tfrecord_lib.convert_to_feature(filename.encode('utf8')),
          'image/format':
              tfrecord_lib.convert_to_feature('jpg'.encode('utf8')),
          'image/height':
              tfrecord_lib.convert_to_feature(image_info['height']),
          'image/width':
              tfrecord_lib.convert_to_feature(image_info['width']),
          'image/source_id':
              tfrecord_lib.convert_to_feature(str(image_info['id']).encode('utf8')),
          'image/object/bbox/xmin':
              tfrecord_lib.convert_to_feature(data['xmin']),
          'image/object/bbox/xmax':
              tfrecord_lib.convert_to_feature(data['xmax']),
          'image/object/bbox/ymin':
              tfrecord_lib.convert_to_feature(data['ymin']),
          'image/object/bbox/ymax':
              tfrecord_lib.convert_to_feature(data['ymax']),
          'image/object/class/text':
              tfrecord_lib.convert_to_feature(data['category_names']),
          'image/object/class/label':
              tfrecord_lib.convert_to_feature(data['category_id']),
          'image/object/is_crowd':
              tfrecord_lib.convert_to_feature(data['is_crowd']),
          'image/object/area':
              tfrecord_lib.convert_to_feature(data['area'], 'float_list'),
          'image/object/mask':
              tfrecord_lib.convert_to_feature(data['encoded_mask_png'])
      }
      # print(keys_to_features['image/object/class/label'])
      example = tf.train.Example(
          features=tf.train.Features(feature=keys_to_features))
      return example



    writers = [
        tf.io.TFRecordWriter(
            str(tf_record_path) + prefix +'-%05d-of-%05d.tfrecord' % (i, num_shards))
        for i in range(num_shards)
    ]
    id_to_name_map = {cat_dict['id']: cat_dict['name']
                      for cat_dict in lvis_annotation.categories}
    
    for idx, image_info in enumerate(tqdm.tqdm(lvis_annotation.images)):
      tf_example = _process_example(images_path, image_info, id_to_name_map)
      writers[idx % num_shards].write(tf_example.SerializeToString())

    del lvis_annotation


def createTFrecord(recordsToPath : Path, imagesPath : Path, annotationPath : Path):
    tfrecords_dir = recordsToPath # Folder for where the tfrecords will go
    images_dir = imagesPath # Images path
    annotation_file = annotationPath # Annotations json
    with open(annotation_file, "r") as f:
        annotations = json.load(f)["annotations"]
    print(f"Number of images found in JSON file: {len(annotations)}")

        
    num_samples = 6
    num_tfrecords = len(annotations) // num_samples
    if len(annotations) % num_samples:
        num_tfrecords += 1  # add one record if there are any remaining samples
    if not Path.exists(tfrecords_dir):
        Path.mkdir(tfrecords_dir)  # creating TFRecords output folder

    # Helper functions
    def image_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
        )

    def bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


    def float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    def int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def float_feature_list(value):
        """Returns a list of float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


    def create_example(image, path, example):
        feature = {
            "image": image_feature(image),
            "path": bytes_feature(path),
            "area": float_feature(example["area"]),
            "bbox": float_feature_list(example["bbox"]),
            "category_id": int64_feature(example["category_id"]),
            "id": int64_feature(example["id"]),
            "image_id": int64_feature(example["image_id"]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))


    def parse_tfrecord_fn(example):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "path": tf.io.FixedLenFeature([], tf.string),
            "area": tf.io.FixedLenFeature([], tf.float32),
            "bbox": tf.io.VarLenFeature(tf.float32),
            "category_id": tf.io.FixedLenFeature([], tf.int64),
            "id": tf.io.FixedLenFeature([], tf.int64),
            "image_id": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
        example["bbox"] = tf.sparse.to_dense(example["bbox"])
        return example  


    
    listImages = sorted(list(imagesPath.iterdir())) # Sort the paths to match the image segmentation
    x = 0


    for tfrec_num in range(num_tfrecords):
        samples = annotations[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]
        with tf.io.TFRecordWriter(
            str(tfrecords_dir) + "/file_00-%i.tfrec" % (len(samples)) # Debugging (00 - %2i)
        ) as writer:
            for sample in samples:
                image_path = f"{images_dir}/{listImages[x].name}"
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                example = create_example(image, image_path, sample)
                writer.write(example.SerializeToString())
                x += 1


def parse_and_displayRecord(tfRecordPath : Path, numToDisplay : int=1):
    if not tfRecordPath.exists():
        raise Exception("The path you provided isn't valid or doesn't exist.") 
    
    def parse_tfrecord_fn(example):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "path": tf.io.FixedLenFeature([], tf.string),
            #"area": tf.io.FixedLenFeature([], tf.float32),
            "bbox": tf.io.VarLenFeature(tf.float32),
            "category_id": tf.io.FixedLenFeature([], tf.int64),
            "id": tf.io.FixedLenFeature([], tf.int64),
            "image_id": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
        example["bbox"] = tf.sparse.to_dense(example["bbox"])
        return example  
    
    raw_dataset = tf.data.TFRecordDataset(tfRecordPath.absolute())
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

    for features in parsed_dataset.take(numToDisplay):
        print("Displaying image:")
        print(f"Image shape: {features['image'].shape}")
        print(f"Bbox shape: {features['bbox']}")
        print(f"Path: {features['path']}")
        plt.figure(figsize=(7, 7))
        plt.imshow(features["image"].numpy())
        plt.show()

    
