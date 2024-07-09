import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import matplotlib.pyplot as plt

def getCoord(joint):
    return (joint.x, joint.y, joint.z)

def plotBodyPart(ax, joint1, joint2):
    x1, y1, z1 = getCoord(joint1)
    x2, y2, z2 = getCoord(joint2)

    ax.plot([x1, x2], [y1, y2], [z1, z2])
    
def plotBody(ax, pose_landmarks):
    pls = pose_landmarks

    #plot collarbone

    plotBodyPart(ax, pls[11], pls[12])

    #plot arms

    plotBodyPart(ax, pls[11], pls[13]) #left upper arm
    plotBodyPart(ax, pls[13], pls[15]) #left forearm

    plotBodyPart(ax, pls[12], pls[14]) #right upper arm
    plotBodyPart(ax, pls[14], pls[16]) #right forearm

    #plot spine

    
    x1, y1, z1 = tuple((a + b) / 2 for a, b in zip(getCoord(pls[11]), getCoord(pls[12])))
    x2, y2, z2 = tuple((a + b) / 2 for a, b in zip(getCoord(pls[23]), getCoord(pls[24])))

    ax.plot([x1, x2], [y1, y2], [z1, z2])
    

    #plot hips

    plotBodyPart(ax, pls[23], pls[24])

    #plot legs

    plotBodyPart(ax, pls[23], pls[25]) #left femur
    plotBodyPart(ax, pls[25], pls[27]) #left calf

    plotBodyPart(ax, pls[24], pls[26]) #right femur
    plotBodyPart(ax, pls[26], pls[28]) #right calf

    






fig = plt.figure()
ax = fig.add_subplot(projection='3d')

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')


model_path = "pose_landmarker_lite.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

image = mp.Image.create_from_file("RonnieColeman.jpg")
detection_result = detector.detect(image)
pose_landmarks_list = detection_result.pose_landmarks[0]

image2 = mp.Image.create_from_file("ArnoldSchwarzenegger.jpg")
detection_result2 = detector.detect(image2)
pose_landmarks_list2 = detection_result2.pose_landmarks[0]

plotBody(ax, pose_landmarks_list)
plotBody(ax2, pose_landmarks_list2)

plt.isinteractive = False

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(-1,1)

ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.set_zlim(-1,1)

plt.savefig("testplots/RonnieColemanPlot.png")

plt.show()