import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import commonroad_prediction.HelperFunctions as HelperFunctions


def orientVectorListTorch(vectorList):
    orientVectorList = torch.zeros((30, 2))
    relativeAngle = torch.atan2(vectorList[0][1], vectorList[0][0])
    orientVectorList[0][0] = vectorList[0][0]
    orientVectorList[0][1] = vectorList[0][1]

    for i in range(len(vectorList)-1):
        dx = vectorList[i+1][0]
        dy = vectorList[i+1][1]
        if torch.abs(dx) < 1e-8 and torch.abs(dy) < 1e-8:
            angle = torch.tensor(0.0)
        else:
            angle = torch.atan2(dy, dx)
        relativeAngle = angle + relativeAngle

        orientVectorList[i+1][0] = dx * torch.cos(relativeAngle) - dy * torch.sin(relativeAngle)
        orientVectorList[i+1][1] = dx * torch.sin(relativeAngle) + dy * torch.cos(relativeAngle)

    return orientVectorList

def convertingToAbsolutePointListTorch(orientVectorList):
    x_last = torch.tensor(0.0)
    y_last = torch.tensor(0.0)
    x = x_last
    y = y_last
    pointList = torch.zeros((30, 2))

    for i in range(len(orientVectorList)):
        x_last, y_last = x, y
        x, y = x_last + orientVectorList[i][0], y_last + orientVectorList[i][1]
        pointList[i][0] = x
        pointList[i][1] = y
    return pointList

def convertRelativeDxDyTensorAbsolutePointListTorch(relativeDxDyList):
    relativeDxDyWithOrientetAnglesList = orientVectorListTorch(relativeDxDyList.view(30, 2))
    orientedPointList = convertingToAbsolutePointListTorch(relativeDxDyWithOrientetAnglesList)
    return orientedPointList


def orientVectorList(vectorList):
    orientVectorList = np.zeros((30, 2))
    relativeAngle = np.arctan2(vectorList[0][1], vectorList[0][0]) 
    orientVectorList[0][0] = vectorList[0][0] 
    orientVectorList[0][1] = vectorList[0][1] 

    for i in range(len(vectorList)-1):
        dx = vectorList[i+1][0]
        dy = vectorList[i+1][1]
        if np.abs(dx) < 1e-8 and np.abs(dy) < 1e-8:  # Check for very small values
            angle = 0.0  # Treat near-zero displacements as zero angle
        else:
            angle = np.arctan2(dy, dx) # ToDo: check if this is correct
        relativeAngle = angle + relativeAngle
        # relativeAngle = np.arctan2(np.sin(angle + relativeAngle), np.cos(angle + relativeAngle))

        orientVectorList[i+1][0] = dx * np.cos(relativeAngle) - dy * np.sin(relativeAngle)
        orientVectorList[i+1][1] = dx * np.sin(relativeAngle) + dy * np.cos(relativeAngle)

    return orientVectorList
    
def convertingToAbsolutePointList(orientVectorList):
    x_last = 0.0
    y_last = 0.0
    x = x_last
    y = y_last
    pointList = np.zeros((30, 2))

    # Plot each x and y pair on the axis
    for i in range(len(orientVectorList)):
        x_last, y_last = x, y
        x, y = x_last + orientVectorList[i][0], y_last + orientVectorList[i][1]
        pointList[i][0] = x
        pointList[i][1] = y
    return pointList

def convertRelativeDxDyTensorAbsolutePointList(relativeDxDyList):
    relativeDxDyWithOrientetAnglesList = orientVectorList(relativeDxDyList.view(30, 2).tolist())
    orientedPointList = convertingToAbsolutePointList(relativeDxDyWithOrientetAnglesList)
    return orientedPointList

def convertLocalPointListToGlobalPointList(localPointList, globalX, globalY, globalOrientation): # ToDo: check if this !!!
    globalPointList = np.zeros((30, 2))
    for i in range(len(localPointList)):
        globalPointList[i][0] = globalX + localPointList[i][0] * np.cos(globalOrientation) - localPointList[i][1] * np.sin(globalOrientation)
        globalPointList[i][1] = globalY + localPointList[i][0] * np.sin(globalOrientation) + localPointList[i][1] * np.cos(globalOrientation)
    return globalPointList
    
def getOrientationListAndVelocityList(vectorList, initialOrientation, dt):
    vectorList = vectorList.view(30, 2).tolist()
    globalOrientationList = np.zeros((30))
    velocityList = np.zeros((30))
    velocityList[0] = HelperFunctions.calculateVelocity(vectorList[0][0], vectorList[0][1], dt)
    relativeAngle = initialOrientation 
    globalOrientationList[0] = relativeAngle
    print(vectorList)
    for i in range(len(globalOrientationList)-1):
        dx = vectorList[i+1][0]
        dy = vectorList[i+1][1]
        velocityList[i+1] = HelperFunctions.calculateVelocity(dx, dy, dt)
        # if (dx <= 0.00001):
        #     angle = np.arctan2(0, 1) 
        # else:
        angle = np.arctan2(dy, dx) # ToDo: check if this is correct
        relativeAngle = relativeAngle + angle 
        if relativeAngle > np.pi:
            relativeAngle -= 2 * np.pi
        elif relativeAngle < -np.pi:
            relativeAngle += 2 * np.pi
        globalOrientationList[i+1] = relativeAngle

    return globalOrientationList, velocityList    

def convertOutputTrajectoryToGloabalPointList(output, cr_id, groundTruthPositionForCurrentTimeStep,viewLength):
    localPointList = convertRelativeDxDyTensorAbsolutePointList(output)
    localPointList = getPointListInMapBoundary(localPointList, viewLength)
    globalX, globalY = groundTruthPositionForCurrentTimeStep[cr_id]['position'][0], groundTruthPositionForCurrentTimeStep[cr_id]['position'][1]
    globalOrientation = groundTruthPositionForCurrentTimeStep[cr_id]['orientation']
    globalPointList = convertLocalPointListToGlobalPointList(localPointList, globalX, globalY, globalOrientation)
    return globalPointList


def outputTrajectoryOnDrivableRoadScalar(outputPointList, mapInView):
    input_sizeX = 39
    gtLength = 30
    metaFactor = (input_sizeX)/27
    mapInView = np.flipud(mapInView)

    outputPointList = convertRelativeDxDyTensorAbsolutePointList(outputPointList)
    rotation_matrix = np.array([[0, -1], [1, 0]]) # Rotate the vectors counterclockwise by 90 degrees
    # outputPointList = np.dot(outputPointList.view(30,2).detach().numpy(), rotation_matrix.T)
    outputPointList = np.dot(outputPointList, rotation_matrix.T)
    overlapScalar = 0
    outsideBoundary = 0
    for i in range(gtLength):
        if np.isnan(outputPointList[i][0]).any() or np.isnan(outputPointList[i][1]).any(): 
            # print('output has nan')
            if(i-outsideBoundary == 0): return 0.1
            return max(overlapScalar/(i-outsideBoundary),0.05)

        matrixY = int(round(outputPointList[i][0]*metaFactor) + int((input_sizeX/2) -1))
        matrixX = int(round(outputPointList[i][1]*metaFactor))
        if(matrixX >= 0 and matrixX < input_sizeX-1 and matrixY >= 0 and matrixY < input_sizeX-1):
            if(matrixX == 0 or matrixX == input_sizeX-1 or matrixY == 0 or matrixY == input_sizeX-1): 
                if(mapInView[matrixX][matrixY] == 1):
                    overlapScalar += 1
                else:   
                    overlapScalar += 0
            else:
                activation = 0 # This is a 3x3 kernel with activation in the middle and the corners [0.1 x 0 x 0.1] | [0 x 4.6 x 0] | [0.1 x 0 x 0.1]
                if(mapInView[matrixX][matrixY] == 1):
                    activation = activation + 4.0       
                if(mapInView[matrixX-1][matrixY-1] == 1):
                    activation = activation + 0.25        
                if(mapInView[matrixX-1][matrixY+1] == 1):
                    activation = activation + 0.25       
                if(mapInView[matrixX+1][matrixY-1] == 1):
                    activation = activation + 0.25        
                if(mapInView[matrixX+1][matrixY+1] == 1):
                    activation = activation + 0.25   
                overlapScalar += activation/5    
        else:
            outsideBoundary += 1 # outside the 27x27m boundary  
    lengthInsideBoundary = (gtLength-outsideBoundary)    
    if(lengthInsideBoundary != 0):
        return max(overlapScalar/lengthInsideBoundary, 0.1)
    return max(overlapScalar/0.1,0.1)

def getPointListInMapBoundary(pointList,viewLength):
    """
    Enforce boundary constraints on  point coordinates. If prediction out of mapInView distance or, the last predictions are set to the previous value that is in the dimensions.

    Args:
    pointList (list or array): List of point coordinates [(x, y), ...].
    viewLength (float): mapInView distance.

    Returns:
    list or array: Modified point coordinates respecting boundary.
    """
    index = 0
    print(f'globalPointListShape0 =  {pointList.shape[0]}')
    for i in range(pointList.shape[0]):
        index = i
        if pointList[i][0] > viewLength or abs(pointList[i][1]) > viewLength/2: break
    while index < pointList.shape[0]:
        pointList[index][0] = pointList[i-1][0]
        pointList[index][1] = pointList[i-1][1]
        index += 1    
    return pointList
        