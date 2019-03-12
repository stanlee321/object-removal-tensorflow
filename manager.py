import multiprocessing
import numpy as np
import cv2
from model import DeepModel



class InputManager:
    """
    This class is used for manage the input procedure for the multiprocessing case.

    ::Args:
    ::model: Name of the model, Type String
    ::multi: Type Boolean,  If True, this will create a parallel worker for the model, 
    
    __init__(model='seg_obj', multi=False)
    
    """
    def __init__(self, model='seg_obj', multi=False):
        # 0 for simple, 1 for multi
        if multi == True:
            _mode = 1
        else:
            _mode = 0

        self.mode = _mode

        if model == 'seg_obj':
            self.deeplab_model = DeepModel(obj=15) 
        else:
            # TODO, make a set of load model checker
            pass
        # initialize the input queue (frames), output queue (detections),
        # and the list of actual detections returned by the child process
        if self.mode == 1:
            self.inputQueue = multiprocessing.Queue(maxsize=1)
            self.outputQueue = multiprocessing.Queue(maxsize=1)
            
            # Instantiate Parallel Process
            self.parallelModel = ParallelProcessManager(self.deeplab_model, self.inputQueue, self.outputQueue)
            
            # construct a child process *indepedent* from our main process of
            # execution
            print("[INFO] starting process...")
            # p = Process(target=detect_on_frame, args=(model, out_frame, in_detect,))
            self.parallelModel.daemon = True
            self.parallelModel.start()

    def predict(self, image_np):
        # if the input queue *is* empty, give the current frame to
        # classify
        if self.inputQueue.empty():
            self.inputQueue.put(image_np)
        # if the output queue *is not* empty, grab the detections
        if not self.outputQueue.empty():
            seg_map = self.outputQueue.get()
        else:
            seg_map = None
        return seg_map

    def terminate(self):
        if self.mode ==1:
            self.parallelModel.terminate()
            self.parallelModel.join()
        else:
            pass
    

class ParallelProcessManager(multiprocessing.Process):
    """
    This class is used for instantiate a prcess in parallel, 
    this has inputQueue and outputQueue, for receive and send the results
    of detections on model
    """
    def __init__(self, model, inputQueue,outputQueue):
        super(ParallelProcessManager, self).__init__()
        self.model = model
        self.net = self.model.forward()

        self.inputQueue = inputQueue
        self.outputQueue = outputQueue
    
    def read_image(self, img):
        
        w, h, _ = img.shape

        ratio = 512. / np.max([w,h])
        image_np = cv2.resize(img, (int(ratio*h),int(ratio*w)))
        resized = image_np / 127.5 - 1.
        pad_x = int(512 - resized.shape[0])
        resized_im = np.pad(resized, ((0, pad_x),(0,0),(0,0)), mode='constant')
        return img, resized_im, pad_x

    def run(self):
        # Make the detection part in another process
        # keep looping
        while True:

            # check to see if there is a frame in our input queue
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            image_np = self.inputQueue.get()

            image, resized_im, _ = self.read_image(image_np)
            self.model.mask_model.create_mask_tool.image = image


            seg_map = self.net.predict(np.expand_dims(resized_im, 0))

            # set the blob as input to our deep learning object
            # detector and obtain the detections
            # write the detections to the output queue
            self.outputQueue.put(seg_map)

