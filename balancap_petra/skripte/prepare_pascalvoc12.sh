DATASET_DIR=/home/pmarce/datasets/VOCdevkit/VOC2012/
OUTPUT_DIR=$DATASET_DIR/train_tfrecords
python ../tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2012_train\
    --output_dir=${OUTPUT_DIR}
