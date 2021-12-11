# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import numpy as np
import cv2
import tqdm
import uuid
import shutil
import json
from pdf2image import convert_from_path
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import convert_PIL_to_numpy

from detection import ObjectDetection
from predictor import VisualizationDemo
from detectron2.data import MetadataCatalog

MetadataCatalog.get("dla_val").thing_classes = ['text', 'title', 'list', 'table', 'figure']

# constants
WINDOW_NAME = "COCO detections"
datas=[]
objDatas=[]

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg




def convertPdfToPngPerPage(pdfPath):
    images = convert_from_path(pdfPath)
    return images



def createMetaData(predictions, image, documentName, page):
    objectDetection = ObjectDetection()
    predictions = predictions["instances"].to(demo.cpu_device)
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None

    for index, item in enumerate(classes):
        if item == 4:
            data = {}
            obj={}
            box = list(boxes)[index].detach().cpu().numpy()
            # Crop the PIL image using predicted box coordinates
            img_id=uuid.uuid4()
            crop_img = crop_object(image, box)
            img_path = "./output/{}.jpg".format(img_id)
            crop_img.save(img_path)
            result,className=objectDetection.detect(img_path)
            #objBoxes=result.pred_boxes if result.has("pred_boxes") else None
            #objScores=result.scores if result.has("scores") else None
            objClasses=result.pred_classes.tolist() if result.has("pred_classes") else None
            objLabels=list(map(lambda x: className[x], objClasses))

            """
            obj['boxes']=objBoxes
            obj['scores'] = objScores
            obj['classes'] = objClasses
            obj['lables'] = objLabels
            """


            """
            print("pdf name: ",documentName)
            print("page: ",page)
            print("image id : ",img_id)
            print("position : " ,boxes.tensor[index].numpy())
            print("score : ",scores[index].numpy())
            print("width:",crop_img.width,"px")
            print("height:",crop_img.height,"px")
            """
            data['pdfName'] = documentName
            data['page'] = page
            data['image_id'] = str(img_id)
            data['position'] = boxes.tensor[index].numpy()
            data['score'] = scores[index].numpy()
            data['width'] = crop_img.width
            data['height'] = crop_img.height
            data['objects']= objLabels
            datas.append(data)


def crop_object(image, box):
  """Crops an object in an image

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
  """

  x_top_left = box[0]
  y_top_left = box[1]
  x_bottom_right = box[2]
  y_bottom_right = box[3]
  x_center = (x_top_left + x_bottom_right) / 2
  y_center = (y_top_left + y_bottom_right) / 2

  crop_img = image.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))

  return crop_img



def clearOutputFolder():
    folder = './output'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))








def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    # TODO title,text,list,figure tespit etmek yerine sadece figure tespit etmesini ayarla
    # TODO detectObjelerin skor ve boxes bilgileri json dosyada tutulacak
    # TODO detection sınıfında cv2 ile image dosyadan okumak yerine parametre olarak gönderilecek
    # TODO koddaki gereksiz methodlar kaldırılacak

    if args.input:
        clearOutputFolder()
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        for path in tqdm.tqdm(args.input, disable=not args.output):
            fullPath, documentName = os.path.split(path)

            images = convertPdfToPngPerPage(path)
            for page in range(len(images)):
                #Save pages as images in the pdf
                #images[i].save('page' + str(i) + '.jpg', 'JPEG')

                img= convert_PIL_to_numpy(images[page], format="BGR")
                #img = read_image(path, format="BGR")
                start_time = time.time()
                #predictions, visualized_output= demo.run_on_image(img)
                predictions = demo.run_on_image(img)

                createMetaData(predictions, images[page], documentName, page+1)
                logger.info(
                    "{} , page {} :  detected {} instances in {:.2f}s".format(
                        path,page+1, len(predictions["instances"]), time.time() - start_time
                    )
                )

            """
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
            """

        json_data = json.dumps(datas,cls=NumpyEncoder)
        with open('./metadata/metadata.json', 'w') as f:
            f.write(json_data)

    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
