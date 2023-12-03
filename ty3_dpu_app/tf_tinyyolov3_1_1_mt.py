
import os
import cv2
import colorsys
import random
import numpy as np
from dnndk import n2cube
import time
#import concurrent.futures
#from multiprocessing import Manager
import threading
import queue
from imutils.video import FPS


'''resize image with unchanged aspect ratio using padding'''
def letterbox_image(image, size):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    print(scale)
    
    nw = int(iw*scale)
    nh = int(ih*scale)
    print(nw)
    print(nh)

    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h,w,3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image

'''image preprocessing'''
def pre_process(image, model_image_size):
    image = image[...,::-1]
    image_h, image_w, _ = image.shape
 
    if model_image_size != (None, None):
        assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) 	
    return image_data

def pre_process_parallel(images, model_image_size, max_workers=4):
    # Parallelize the pre-process function using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(pre_process, img, model_image_size) for img in images]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    return results
'''Get model classification information'''	
def get_class(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

'''Get model anchors value'''
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)
    
def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    nu = num_classes + 5
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis = -1)
    grid = np.array(grid, dtype=np.float32)

    box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
    box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
    box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
    return box_xy, box_wh, box_confidence, box_class_probs
	
def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, dtype = np.float32)
    image_shape = np.array(image_shape, dtype = np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxes[..., 0:1],
        box_maxes[..., 1:2]
    ], axis = -1)
    boxes *= np.concatenate([image_shape, image_shape], axis = -1)
    return boxes
	
def boxes_and_scores(feats, anchors, classes_num, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = _get_feats(feats, anchors, classes_num, input_shape)
    boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, [-1, classes_num])
    return boxes, box_scores	

def draw_bbox(image, bboxes, classes):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        predicted_class = classes[class_ind]
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        
        # Add text with predicted class
        label = "{} {:.2f}".format(predicted_class, score)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 2)[0]
        c2 = c1[0] + text_size[0], c1[1] - text_size[1] - 5
        cv2.rectangle(image, c1, (int(c2[0]), int(c2[1])), bbox_color, -1)  # filled rectangle for text background
        cv2.putText(image, label, (c1[0], int(c1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), 2, cv2.LINE_AA)
    return image

	
def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.45)[0]  # threshold
        order = order[inds + 1]

    return keep	
  
'''Model post-processing'''
def eval(yolo_outputs, image_shape,  class_names, anchors, max_boxes = 80):
    score_thresh = 0.2
    nms_thresh = 0.45
    #class_names = get_class(classes_path)
    #anchors     = get_anchors(anchors_path)
    anchor_mask = [[3, 4, 5], [0, 1, 2]]
    boxes = []
    box_scores = []
    
    input_shape = np.shape(yolo_outputs[0])[1 : 3]
    print(input_shape)
    input_shape = np.array(input_shape)*32
    print(input_shape)
    
    for i in range(len(yolo_outputs)):
        _boxes, _box_scores = boxes_and_scores(yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis = 0)
    box_scores = np.concatenate(box_scores, axis = 0)
    
    mask = box_scores >= score_thresh
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(len(class_names)):
        class_boxes_np = boxes[mask[:, c]]
        class_box_scores_np = box_scores[:, c]
        class_box_scores_np = class_box_scores_np[mask[:, c]]
        nms_index_np = nms_boxes(class_boxes_np, class_box_scores_np) 
        class_boxes_np = class_boxes_np[nms_index_np]
        class_box_scores_np = class_box_scores_np[nms_index_np]
        classes_np = np.ones_like(class_box_scores_np, dtype = np.int32) * c
        boxes_.append(class_boxes_np)
        scores_.append(class_box_scores_np)
        classes_.append(classes_np)
    boxes_ = np.concatenate(boxes_, axis = 0)
    scores_ = np.concatenate(scores_, axis = 0)
    classes_ = np.concatenate(classes_, axis = 0)
    
    return boxes_, scores_, classes_

def infer_image(image_path, task, class_names, anchors, input_shape=(416, 416)):
    # Load image
    loop_start = time.time()
    image = cv2.imread(image_path)
    image_ho, image_wo, _ = image.shape
    image_size = image.shape[:2]

    # Preprocess image
    pre_process_start = time.time()
    image_data = pre_process(image, input_shape)
    pre_process_elapsed = time.time() - pre_process_start

    image_data = np.array(image_data, dtype=np.float32)

    hw_start = time.time()
    # Set input tensor
    input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)
    n2cube.dpuSetInputTensorInHWCFP32(task, CONV_INPUT_NODE, image_data, input_len)

    # Run the DPU task
    n2cube.dpuRunTask(task)

    # Get the output tensors


    conv_out1 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE1, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE1))
    conv_out2 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE2, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE2))
    #conv_out3 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE3, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE3))
    hw_elapsed = time.time() - hw_start

    post_process_start = time.time()
    conv_out1 = np.reshape(conv_out1, (1, 13, 13, 255))
    conv_out2 = np.reshape(conv_out2, (1, 26, 26, 255))
    #conv_out3 = np.reshape(conv_out3, (1, 52, 52, 75))

    yolo_outputs = [conv_out1, conv_out2]

    # Perform post-processing

    out_boxes, out_scores, out_classes = eval(yolo_outputs, image_size, class_names, anchors)


    # Draw bounding boxes on the image
    items = []
    draws = []
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_ho, np.floor(bottom + 0.5).astype('int32'))
        right = min(image_wo, np.floor(right + 0.5).astype('int32'))
        draw = [left, top, right, bottom, score, c]
        item = [predicted_class, score, left, top, right, bottom]
        draws.append(draw)
        items.append(item)
    post_process_elapsed = time.time() - post_process_start


    # Draw bounding boxes on the image
    image_result = draw_bbox(image, draws, class_names)
    loop_elapsed = time.time() - loop_start

    result_image_name = os.path.splitext(os.path.basename(image_path))[0] + "_result.jpg"
    result_image_path = os.path.join("./", result_image_name)
    cv2.imwrite(result_image_path, image_result)

    return items, (pre_process_elapsed, hw_elapsed, post_process_elapsed, loop_elapsed)

def infer_video(video_path, task, class_names, anchors, input_shape=(416, 416)):
    cap = cv2.VideoCapture(video_path)
    video_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, video_size)

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_size = frame.shape[:2]
        image_data = pre_process(frame, input_shape)
        image_data = np.array(image_data, dtype=np.float32)

        # Set input tensor
        input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)
        n2cube.dpuSetInputTensorInHWCFP32(task, CONV_INPUT_NODE, image_data, input_len)

        # Run the DPU task
        n2cube.dpuRunTask(task)

        # Get the output tensors
        conv_out1 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE1, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE1))
        conv_out2 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE2, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE2))
       # conv_out3 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE3, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE3))

        conv_out1 = np.reshape(conv_out1, (1, 13, 13, 255))
        conv_out2 = np.reshape(conv_out2, (1, 26, 26, 255))
        #conv_out3 = np.reshape(conv_out3, (1, 52, 52, 75))

        yolo_outputs = [conv_out1, conv_out2]

        # Perform post-processing
        out_boxes, out_scores, out_classes = eval(yolo_outputs, image_size, class_names, anchors)

        # Draw bounding boxes and labels on the frame
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image_size[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image_size[1], np.floor(right + 0.5).astype('int32'))

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw label and score
            label = "{} {:.2f}".format(predicted_class, score)
            cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Save the annotated frame
            # if(len(out_boxes)) > 0:
            
        frame_count += 1
        #if frame_count % 30 == 0:  # Print FPS every 30 frames
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        fps_text = "FPS: {:.2f}".format(fps)
        print("FPS: {:.2f}".format(fps))
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)


    cap.release()
    out.release()

# functions with clear CPU vs DPU distinction
def pre_process_video_frame(frame, input_shape):
    # Process a single frame
    image_size = frame.shape[:2]
    image_data = pre_process(frame, input_shape)
    image_data = np.array(image_data, dtype=np.float32)
    return image_data, image_size




def dpu_process_video_frame(task, image_data):
    input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)
    n2cube.dpuSetInputTensorInHWCFP32(task, CONV_INPUT_NODE, image_data, input_len)

    # Run the DPU task
    n2cube.dpuRunTask(task)

    # Get the output tensors
    conv_out1 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE1, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE1))
    conv_out2 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE2, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE2))
    return conv_out1,conv_out2

def post_process_video_frame(frame, conv_out1, conv_out2, class_names, anchors, image_size, input_shape):
    conv_out1 = np.reshape(conv_out1, (1, 13, 13, 255))
    conv_out2 = np.reshape(conv_out2, (1, 26, 26, 255))
    #conv_out3 = np.reshape(conv_out3, (1, 52, 52, 75))

    yolo_outputs = [conv_out1, conv_out2]

    # Perform post-processing
    out_boxes, out_scores, out_classes = eval(yolo_outputs, image_size, class_names, anchors)

    # Draw bounding boxes and labels on the frame
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_size[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(image_size[1], np.floor(right + 0.5).astype('int32'))

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label and score
        label = "{} {:.2f}".format(predicted_class, score)
        cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def process_video_serial(video_path, task, class_names, anchors, input_shape=(416,416)):
    cap = cv2.VideoCapture(video_path)
    video_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, video_size)

    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_data, image_size = pre_process_video_frame(frame, input_shape)
        conv_out1, conv_out2 = dpu_process_video_frame(task, image_data)
        frame = post_process_video_frame(frame, conv_out1, conv_out2, class_names, anchors, image_size, input_shape)

        frame_count += 1
        #if frame_count % 30 == 0:  # Print FPS every 30 frames
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        fps_text = "FPS: {:.2f}".format(fps)
        print("FPS: {:.2f}".format(fps))
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)


    cap.release()
    out.release()

global bQuit

def taskvideoCapture(cap, queueIn, inputId=0, input_shape=(416,416)):

    global bQuit

    print("[INFO] taskCapture : starting thread ...")

    # Start the FPS counter
    #fpsIn = FPS().start()

    #frame_count = 0
    #start_time = time.time()
    
    while not bQuit and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #fpsIn.update()
        image_data, image_size = pre_process_video_frame(frame, input_shape)
        queueIn.put((image_data,image_size,frame))
    #fpsIn.stop()
    #print("[INFO] taskCapture : elapsed time: {:.2f}".format(fpsIn.elapsed()))
    #print("[INFO] taskCapture : elapsed FPS: {:.2f}".format(fpsIn.fps()))

    print("[INFO] taskCapture : exiting thread ...")
def taskWorker(cap,task,queueIn,queueOut):

    global bQuit

    print("[INFO] taskWorker : starting thread ...")
    while queueIn.empty():
        print("DPU Input queue empty..waiting..")
        
    while not bQuit and not queueIn.empty():
        # Pop captured image from input queue
        image_data, image_size, frame = queueIn.get()

        # Vitis-AI/DPU based face detector
        conv_out1,conv_out2 = dpu_process_video_frame(task, image_data)

        queueOut.put((conv_out1,conv_out2, frame,image_size))


    # workaround : to ensure other worker threads stop, 
    #              make sure input queue is not empty 
    #queueIn.put((image_data, image_size, frame))

def taskDisplay(cap, out, queueOut, class_names, anchors, input_shape=(416,416)):

    global bQuit

    print("[INFO] taskDisplay : starting thread ...")

    # Start the FPS counter
    frame_count = 0
    start_time = time.time()

    while not bQuit:
        # Pop processed image from output queue
        try:
            conv_out1,conv_out2, frame ,image_size= queueOut.get(timeout=2)
            frame = post_process_video_frame(frame, conv_out1, conv_out2, class_names, anchors, image_size, input_shape)

            # Update the FPS counter
            frame_count +=1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            fps_text = "FPS: {:.2f}".format(fps)
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(frame)

            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        except queue.Empty:
            bQuit=True

        
    # Trigger all threads to stop
    print("Finishing")
    bQuit = True
    cap.release()
    out.release()


    # Stop the timer and display FPS information

   # print("[INFO] taskDisplay : elapsed time: {:.2f}".format(fpsOut.elapsed()))
    # print("[INFO] taskDisplay : elapsed FPS: {:.2f}".format(fpsOut.fps()))

    # Cleanup
    cv2.destroyAllWindows()

    print("[INFO] taskDisplay : exiting thread ...")

    print("[INFO] taskWorker : exiting thread ...")
def process_video_parallel(video_path, task, class_names, anchors, input_shape=(416,416)):
    global bQuit
    bQuit = False
    cap = cv2.VideoCapture(video_path)
    video_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, video_size)
    queueIn = queue.Queue()
    queueOut = queue.Queue()
    threads = 4
        # Launch threads
    threadAll = []
    tc = threading.Thread(target=taskvideoCapture, args=(cap, queueIn))
    threadAll.append(tc)
    for i in range(threads):
        tw = threading.Thread(target=taskWorker, args=(cap,task,queueIn,queueOut))
        threadAll.append(tw)
    td = threading.Thread(target=taskDisplay, args=(cap, out, queueOut, class_names, anchors))
    threadAll.append(td)
    for x in threadAll:
        x.start()

    # Wait for all threads to stop
    for x in threadAll:
        x.join()


def process_frame(frame, task, class_names, anchors, input_shape):
    # Process a single frame
    image_size = frame.shape[:2]
    image_data = pre_process(frame, input_shape)
    image_data = np.array(image_data, dtype=np.float32)

    # Set input tensor
    input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)
    n2cube.dpuSetInputTensorInHWCFP32(task, CONV_INPUT_NODE, image_data, input_len)

    # Run the DPU task
    n2cube.dpuRunTask(task)

    # Get the output tensors
    conv_out1 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE1, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE1))
    conv_out2 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE2, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE2))
    #conv_out3 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE3, n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE3))

    conv_out1 = np.reshape(conv_out1, (1, 13, 13, 255))
    conv_out2 = np.reshape(conv_out2, (1, 26, 26, 255))
    #conv_out3 = np.reshape(conv_out3, (1, 52, 52, 75))

    yolo_outputs = [conv_out1, conv_out2]

    # Perform post-processing
    out_boxes, out_scores, out_classes = eval(yolo_outputs, image_size, class_names, anchors)

    # Draw bounding boxes and labels on the frame
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_size[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(image_size[1], np.floor(right + 0.5).astype('int32'))

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label and score
        label = "{} {:.2f}".format(predicted_class, score)
        cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# def process_video_parallel(video_path, task, class_names, anchors, input_shape=(416, 416), num_workers=2):
#     cap = cv2.VideoCapture(video_path)
#     video_size = (int(cap.get(3)), int(cap.get(4)))
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('output.avi', fourcc, 20.0, video_size)

#     frame_count = 0
#     start_time = time.time()

#     frames_queue = Queue()

#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frames_queue.put(frame)
#             future = executor.submit(process_frame, frames_queue.get(), task, class_names, anchors, input_shape)
#             processed_frame = future.result()
            
#             frame_count += 1     
#             elapsed_time = time.time() - start_time
#             fps = frame_count / elapsed_time
#             fps_text = "FPS: {:.2f}".format(fps)
#             cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             out.write(processed_frame)
#             print("FPS: {:.2f}".format(fps))

#     cap.release()
#     out.release()

# def infer_video_parallel(video_path, task, class_names, anchors, input_shape=(416, 416)):
#     cap = cv2.VideoCapture(video_path)
#     video_size = (int(cap.get(3)), int(cap.get(4)))
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('output.avi', fourcc, 20.0, video_size)

#     frame_count = 0
#     start_time = time.time()

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # Process frames in parallel
#         futures = []
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             futures.append(executor.submit(process_frame, frame.copy(), task, class_names, anchors, input_shape))

#         # Collect results
#         for future in concurrent.futures.as_completed(futures):
#             frame_count += 1
#             #if frame_count % 30 == 0:  # Print FPS every 30 frames
#             elapsed_time = time.time() - start_time
#             fps = frame_count / elapsed_time
#             fps_text = "FPS: {:.2f}".format(fps)
#             print("FPS: {:.2f}".format(fps))
#             annotated_frame = future.result()
#             cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             out.write(annotated_frame)



#     cap.release()
#     out.release()

def infer_camera(task, class_names, anchors, input_shape=(416, 416), save_video=True):
    cap = cv2.VideoCapture(0)  # Use the appropriate camera index, e.g., 0 or 1

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    video_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_camera.avi', fourcc, 20.0, video_size)

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally for a more natural view

        # Process the frame
        annotated_frame = process_frame(frame, task, class_names, anchors, input_shape)

        # Display the annotated frame
        cv2.imshow('Object Detection', annotated_frame)     

        frame_count += 1
        #if frame_count % 30 == 0:  # Print FPS every 30 frames
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        fps_text = "FPS: {:.2f}".format(fps)
        print("FPS: {:.2f}".format(fps))
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        # Save the annotated frame to the video file
        if save_video:
            out.write(annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()


"""DPU Kernel Name for tf_yolov3tiny"""
KERNEL_CONV="tf_yolov3tiny"

"""DPU IN/OUT Name for tf_yolov3tiny"""
CONV_INPUT_NODE="yolov3_tiny_convolutional1_Conv2D"
CONV_OUTPUT_NODE1="yolov3_tiny_convolutional10_Conv2D"
CONV_OUTPUT_NODE2="yolov3_tiny_convolutional13_Conv2D"
#CONV_OUTPUT_NODE3="conv2d_75_convolution"

if __name__ == "__main__":
    
    classes_path = "./image/coco_classes.txt"
    #class_names = get_class(classes_path)
    class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

    anchors_path = "./model_data/yolo_anchors.txt"
    #anchors = get_anchors(anchors_path)
    anchor_list = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]
    anchor_list_float = [float(x) for x in anchor_list]
    anchors = np.array(anchor_list_float).reshape(-1, 2)

    available_modes = ["i","v","c"]

    mode_select = "v"
    
    # Image Mode options
    #image_paths = ["/home/root/person.jpg", "/home/root/dog.jpg", "/home/root/eagle.jpg"]
    # image_paths = ["/home/root/dog.jpg",]
    image_paths = ["/home/root/shgpics/gerst1.jpeg","/home/root/shgpics/gerst2.jpg","/home/root/shgpics/sh1.jpg","/home/root/shgpics/snor.PNG","/home/root/shgpics/shsop.JPG" ]
    # Video Mode options
    #video_paths = ["/home/root/video.mp4"]
    video_paths = ["/home/root/traffic-mini.mp4"]

    # Camera Mode options


    main_time_start = time.time()

    if(mode_select == "i"):
        n2cube.dpuOpen()
        kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
        task = n2cube.dpuCreateTask(kernel, 0)


        for image_path in image_paths:
            print("Current Image:", image_path)
            items, time_tuple = infer_image(image_path, task, class_names, anchors)
            print("\nItems: ", items)
            print("\nPre-Process Time :", time_tuple[0], 
                  "\nHW Inference time :",time_tuple[1],
                   "\nPost-Process Time :", time_tuple[2],
                    "\nLoop Time :", time_tuple[3] )

        n2cube.dpuDestroyTask(task)
        n2cube.dpuClose()
        print("Total main time:", time.time() - main_time_start)
            # cv2.imshow("showing",image_result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    elif(mode_select == "v"):
        n2cube.dpuOpen()
        kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
        task = n2cube.dpuCreateTask(kernel, 0)

        for video_path in video_paths:
            print("Current Video:", video_path)
            process_video_parallel(video_path, task, class_names, anchors)

        n2cube.dpuDestroyTask(task)
        n2cube.dpuClose()
        print("Total main time:", time.time() - main_time_start)
    
    elif mode_select == "c":
        n2cube.dpuOpen()
        kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
        task = n2cube.dpuCreateTask(kernel, 0)

        print("Running Camera Inference. Press 'q' to exit.")
        infer_camera(task, class_names, anchors)

        n2cube.dpuDestroyTask(task)
        n2cube.dpuClose()
        print("Total main time:", time.time() - main_time_start)
    
