
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def midpoint(tl, br):
    midpt = (int((tl[0] + br[0])/2), int((tl[1] + br[1])/2))
    return midpt
    




options = {
        
       'model':'cfg/yolo.cfg',
       'load':'bin/yolov2.weights',
       'threshold':0.3,
       'gpu': 1.0
}
tfnet = TFNet(options)

capture = cv2.VideoCapture("video.mp4")
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

frame_width = int( capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int( capture.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('outpy.avi',fourcc, 15, (frame_width,frame_height))


timecount =0
timelist = []
midx = []
midy = []

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            print("{} {}".format(*tl))
            frame = cv2.rectangle(frame, tl, br, color, 7)
            midpt = midpoint(tl, br)
            x = midpt[0]
            y = midpt[1]
            timelist.append(timecount)
            midx.append(x)
            midy.append(y)
            frame = cv2.circle(frame, midpt, int(1), (0,0,255), int(1))
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            out.write(frame)
        cv2.imshow('frame',frame)    
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    timecount = timecount + 1
    print(timecount)
                

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(timelist, midx, midy)
ax.set_xlabel('time')
ax.set_ylabel('xposition')
ax.set_zlabel('yposition')
plt.show()

capture.release()
out.release()
cv2.destroyAllWindows()
print("finished")
