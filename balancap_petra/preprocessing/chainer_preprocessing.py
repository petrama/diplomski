import tensorflow as tf
import random

import tf_extended as tfe


def crop(image, boxes, labels):
    min_obj_covered = random.choice((
        None, 0., 0.1, 0.3, 0.7, 0.9

    ))

    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.expand_dims(boxes, 0),
        min_object_covered=min_obj_covered,
        aspect_ratio_range=(0.5, 2),
        area_range=(0.3, 1),
        max_attempts=50,
        use_image_if_no_bounding_boxes=True)

    distort_bbox = distort_bbox[0, 0]

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    # Restore the shape since the dynamic slice loses 3rd dimension.
    cropped_image.set_shape([None, None, 3])

    # Update bounding boxes: resize and filter out.
    bboxes = tfe.bboxes_resize(distort_bbox, boxes)

    # petra!
    labels, bboxes = tfe.bboxes_filter_center(labels, bboxes)
    bboxes = tfe.bboxes_clip(tf.expand_dims(tf.constant([0., 0., 1., 1.]), 0), bboxes)

    return cropped_image, labels, bboxes


def distort(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if random.randrange(2): image = tf.image.random_brightness(image, max_delta=32. / 255)
    if random.randrange(2): image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    if random.randrange(2): image = tf.image.random_hue(image, max_delta=0.1)
    if random.randrange(2): image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = 255 * tf.clip_by_value(image, 0.0, 1.0)
    image = tf.cast(image, tf.uint8)
    return image

