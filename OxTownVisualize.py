import cv2
import pandas as pd


def main():
    topFilePath = 'dataset/OxfordTown/TownCentre-groundtruth.top'
    vidPath = 'dataset/OxfordTown/TownCentreXVID.mp4'
    df =  pd.read_csv(topFilePath)
    print(df.info())
    frameCount = -4
    cap = cv2.VideoCapture(vidPath)
    if cap.isOpened() == False:
        print('Error loading video')
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frameData = df.loc[df['frameNumber'] == frameCount]
            frameCount += 1
            # print(frameData)
            for i in range (frameData.shape[0]):
                bodyLeft = frameData.iat[i,8]
                bodyTop = frameData.iat[i,9]
                bodyRight = frameData.iat[i,10]
                bodyBottom = frameData.iat[i,11]

                headLeft = frameData.iat[i,4]
                headTop = frameData.iat[i,5]
                headRight = frameData.iat[i,6]
                headBottom = frameData.iat[i,7]
                # print(bodyLeft)
                frame = cv2.rectangle(frame, (int(round(bodyLeft)),int(round(bodyTop))),(int(round(bodyRight)),int(round(bodyBottom))), (255, 0, 0), 1)
                frame = cv2.rectangle(frame, (int(round(headLeft)),int(round(headTop))),(int(round(headRight)),int(round(headBottom))), (0, 255, 0), 1)
            cv2.imshow('Video Strean', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()