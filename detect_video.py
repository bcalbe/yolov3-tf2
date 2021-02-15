import time
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/Town Square_1.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    segment_frame_num = 0
    segment_id = 0
    segment_frame = []
    num_of_obeject  = []
    segment_scores = []
    segment_nums = []
    key_frame = []
    image_id = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./data/video/town_summary.avi',fourcc, 20.0, (240,120))

    while True:
        if segment_frame_num == 20:
            key_num = np.argmax(segment_nums)
            print(key_num)
            #cv2.imwrite("./data/images/{}.jpg".format(image_id),segment_frame[key_num])
            out.write(segment_frame[key_num])
            key_frame.append(segment_frame[key_num])
            segment_scores = []
            segment_frame = []
            segment_nums = []
            segment_frame_num = 0
            image_id += 1

        _, img = vid.read()
        img = cv2.resize(img,(640,480))
        segment_frame.append(img)
        segment_frame_num += 1
   
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        segment_scores.append(scores)
        segment_nums.append(nums)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "Time: {:.2f}ms,num :{}".format(sum(times)/len(times)*1000,nums), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass