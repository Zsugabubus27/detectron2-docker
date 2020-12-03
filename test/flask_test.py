import time
from detectron_client import DetectronClient
import cv2
import imagezmq

class DetectionImageSender(imagezmq.ImageSender):
    def send_image_reqrep(self, msg, image):
        """Sends OpenCV image and msg to hub computer in REQ/REP mode
        Arguments:
          msg: text message or image name.
          image: OpenCV image to send to hub.
        Returns:
          A text reply from hub.
        """
        if image.flags['C_CONTIGUOUS']:
            # if image is already contiguous in memory just send it
            self.zmq_socket.send_array(image, msg, copy=False)
        else:
            # else make it contiguous before sending
            image = np.ascontiguousarray(image)
            self.zmq_socket.send_array(image, msg, copy=False)
        hub_reply = self.zmq_socket.recv_pyobj()  # receive the reply message
        return hub_reply


myFrame1 = cv2.imread("/home/dobreff/videos/samples/1027_14_25_53_first.bmp")
# myFrame2 = cv2.imread("/home/dobreff/videos/samples/1026_05_46_59_first.bmp")
# myFrame3 = cv2.imread("/home/dobreff/videos/samples/1022_07_51_42_first.bmp")

sender = DetectionImageSender(connect_to='tcp://localhost:5556')
print(sender.send_image('Ekkoo', myFrame1))