import cv2
import numpy as np
import utlis

##
path = "ssc.png"
widthImg = 550
heightImg = 700
questions = 25

choices = 4
ans = [1, 3, 0, 1, 3, 2, 1, 1, 3, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
##
img = cv2.imread(path)

# PREPROCESSING
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgFinal = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 150)

# Finding all Contour:
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

# Finding rectangles
rectCon = utlis.rectContour(contours)
biggestContour = utlis.getCornerPoint(rectCon[0])

gradePoints = utlis.getCornerPoint(rectCon[2])
# print(biggestContour)
if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

    biggestContour = utlis.reorder(biggestContour)
    gradePoints = utlis.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [200, 0], [0, 800], [200, 800]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (200, 800))

    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [315, 0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
    # cv2.imshow("Grade Display",imgGradeDisplay)
    cv2.imshow("Eye View", imgWarpColored)

    # Apply Thresholding
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 225, 255, cv2.THRESH_BINARY_INV)[1]

    cv2.imshow("Thresh View", imgThresh)
    boxes = utlis.splitBoxes(imgThresh)
    # cv2.imshow("Test 2",boxes[2])

    # print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))

    myPixelVal = np.zeros((questions, choices))
    countC = 0
    countR = 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if (countC == choices): countR += 1; countC = 0
    # print(myPixelVal)

    myIndex = []
    for x in range(0, questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        # print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    print(myIndex)

    # Gradings
    grading = []
    for x in range(0, questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    print(grading)

    score = (sum(grading) / questions) * 100  # Final GRADE
    print(score)

    # Display grading:
    imgResult = imgWarpColored.copy()
    imgResult = utlis.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
    imRawDrawing = np.zeros_like(imgWarpColored)
    imRawDrawing = utlis.showAnswers(imRawDrawing, myIndex, grading, ans, questions, choices)

    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWarp = cv2.warpPerspective(imRawDrawing, invMatrix, (widthImg, heightImg))
    imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_RGBA2RGB)
    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade, str(int(score)) + "%", (95, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 3)

    InvMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, InvMatrixG, (widthImg, heightImg))

    imgFinal = cv2.addWeighted(imgFinal, 0.7, imgInvWarp, 10, 10)
    imgFinal = cv2.addWeighted(imgFinal, 0.9, imgInvGradeDisplay, 1, 0)

imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBiggestContours, imgBlank, imgBlank])
imgStacked = utlis.stackImages(imageArray, 0.4)

# cv2.imshow("Stacked SSC", imgStacked)
cv2.imshow("Result", imgResult)
cv2.imshow("Main", img)
# cv2.imshow("Draw", imRawDrawing)
cv2.imshow("imgFInal", imgFinal)
# cv2.imshow("Grade", imgRawGrade)

if cv2.waitKey(0) & 0xFF == ord('s'):
    cv2.imwrite("images/FinalResult1.jpg", imgFinal)
    cv2.waitKey(300)
