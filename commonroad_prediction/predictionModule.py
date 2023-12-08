import torch


def makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView):
    """
    Combines encoded map, ego velocity, and social information into a single input tensor as network input.

    Parameters:
    - encodedMap (Tensor): Encoded map information tensor.
    - egoVelocity (float): Velocity of the ego vehicle.
    - socialInformationOfCarsInView (Tensor): Information about surrounding cars within ego vehicle's view.

    Returns:
    - inputTensor (Tensor): Combined tensor containing flattened map, ego velocity, and socialInformationOfCarsInView.
    """
    egoVelocity = torch.tensor(egoVelocity)  # Convert egoVelocity to a tensor
    combined_tensor = torch.cat((encodedMap.view(-1,64), egoVelocity.view(-1,1), socialInformationOfCarsInView.view(-1,4*20)), dim=1) # Concatenate flattened tensors
    # print(combined_tensor.shape)
    inputTensor = combined_tensor
    return inputTensor

def loadNormsForNeuralNet(): 
    egoVelocityNorm = torch.load('commonroad_prediction/NormsTest/eviEgoVelocityNorm39.pt')
    socialInformationOfCarsInViewNorm = torch.load('commonroad_prediction/NormsTest/eviSocialInformationOfCarsInViewNorm39.pt')
    groundTruthNorm = torch.load('commonroad_prediction/NormsTest/eviRelGroundTruthNorm39.pt')
    return egoVelocityNorm, socialInformationOfCarsInViewNorm, groundTruthNorm

def orientVectorListTorch(vectorList):
    dx, dy = vectorList[:,0], vectorList[:,1]
    angle_increment = torch.atan2(dy, dx)
    
    mask = (torch.abs(dx) < 1e-8) & (torch.abs(dy) < 1e-8)
    angle_increment[mask] = 0
    relativeAngle = torch.cumsum(angle_increment, dim=0) 
    relativeAngle[0] = 0
    orientVectorList = torch.stack([
        dx * torch.cos(relativeAngle) - dy * torch.sin(relativeAngle),
        dx * torch.sin(relativeAngle) + dy * torch.cos(relativeAngle)
    ], dim=1)
    return orientVectorList


def convertingToAbsolutePointListTorch(orientVectorList):
    return torch.cumsum(orientVectorList, dim=0)

def convertRelativeDxDyTensorAbsolutePointListTorch(relativeDxDyList):
    """
    Combines the output trajectory from vector list to a point list (from 30 dx,dy --> 30 points)

    Parameters:
    - relativeDxDyList (Tensor): Output trajectory (unnormalized) from neural network in the form of a vector list.
    Returns:
    - output trajectory points (Tensor): Output trajectory (unnormalized) from neural network in the form of a point list.
    """
    relativeDxDyWithOrientetAnglesList = orientVectorListTorch(relativeDxDyList.view(30, 2))
    orientedPointList = convertingToAbsolutePointListTorch(relativeDxDyWithOrientetAnglesList)
    return orientedPointList

def getOutputTrajectoryFromNeuralNet(mapInView, egoVelocity, socialInformationOfCarsInView,learntCNN, model, egoVelocityNorm, socialInformationOfCarsInViewNorm, groundTruthNorm):
    """
    Generate a trajectory prediction for the current car using the given input features.
    Args:
        mapInView (numpy.ndarray): 2D array representing the map information in the car's view.
        egoVelocity (float): The velocity of the current car.
        socialInformationOfCarsInView (numpy.ndarray): 2D array representing the social information of cars in the current car's view.
    Returns:
        numpy.ndarray: The output trajectory prediction for the current car.
    Note:
        - This function utilizes a learned Convolutional Neural Network (CNN) encoder to process the input map.
        - The input map, egoVelocity, and socialInformationOfCarsInView are normalized before processed in the networks
        - Even though present, the groundTruthTrajectory is not accessed this function.
    """
    input_sizeX = mapInView.shape[0]
    input_sizeY = mapInView.shape[1]
    image = torch.tensor(mapInView).float().view(-1, input_sizeX , input_sizeY) 
    image = image.unsqueeze(1)
    encodedMap = learntCNN.encoder(image).view(-1,64) # Output of encoder dimensionality reduction
    egoVelocity = torch.tensor(egoVelocity)
    egoVelocity = torch.where(egoVelocity != 0, egoVelocity / egoVelocityNorm, torch.zeros_like(egoVelocity)) # normalizing the egoVelocity
    socialInformationOfCarsInView = torch.tensor(socialInformationOfCarsInView)
    socialInformationOfCarsInView = torch.where(socialInformationOfCarsInView != 0, socialInformationOfCarsInView / socialInformationOfCarsInViewNorm.view(20,4), torch.zeros_like(socialInformationOfCarsInView)).view(80) # normalizing the socialInformationOfCarsInView
    inputTensor = makeInputTensorValidation(encodedMap, egoVelocity, socialInformationOfCarsInView).view(-1,145).to(torch.float32) # convert to float32
    prediction = model(inputTensor)  # Forward pass -> prediction
    

    ### Postprocessing --> Only positive x-driving direction allowed
    prediction = prediction.view(30,2) # Separating the two columns
    columnx = prediction[:, 0]
    columny = prediction[:, 1]
    
    abs_columnx = torch.abs(columnx) # Computing the absolute values for x column (only positive driving direction allowed) -> more stable training
    prediction = torch.stack((abs_columnx, columny), dim=1).view(30,2) # Combining the results back into a tensor with shape (30, 2)
    prediction = prediction.view(30*2)
    ###

    # prediction is now in the form of 30 dx,dy displacements and is not "unnormalized" to original physical units
    prediction = torch.where(prediction != 0, prediction * groundTruthNorm, torch.zeros_like(prediction)) # denormalizing the output
    # prediction is now in the form of 30 dx,dy displacements in original physical units


    # prediction =convertRelativeDxDyTensorAbsolutePointListTorch(prediction) # if prediction should be in the form of 30 points
    return prediction



learntCNN, model = NeuralNets().loadNeuralNets()
egoVelocityNorm, socialInformationOfCarsInViewNorm, encodedMapNorm, groundTruthNorm = loadNormsForNeuralNet()

