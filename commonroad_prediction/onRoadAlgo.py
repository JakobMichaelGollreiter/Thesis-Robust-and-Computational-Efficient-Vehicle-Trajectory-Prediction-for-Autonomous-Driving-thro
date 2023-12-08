import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def orientVectorListTorch(vectorList):
    dx, dy = vectorList[:,0], vectorList[:,1]
    angle_increment = torch.atan2(dy, dx).to(device)
    
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
    relativeDxDyWithOrientetAnglesList = orientVectorListTorch(relativeDxDyList.view(30, 2))
    orientedPointList = convertingToAbsolutePointListTorch(relativeDxDyWithOrientetAnglesList)
    return orientedPointList


def lengthInsideMapInView(trajectory):
    x,y = trajectory[:,0], trajectory[:,1]
    mask = (x < 0) | (x > 27) | (y < -13.5) | (y > 13.5)
    idx = torch.nonzero(mask)
    return 30 if idx.numel() == 0 else idx[0].item() + 1


def chamfLoss(trajectoyPoints, mapPoints):
    # trajectoyPoints (N, 2)
    # mapPoints (M, 2)
    
    assert torch.is_tensor(trajectoyPoints) and torch.is_tensor(mapPoints)
    dists = (trajectoyPoints[:, None] - mapPoints[None])**2  # (N, M, 2)
    dists = torch.sqrt(torch.sum(dists, axis=-1))  # (N, M)
    dists = torch.min(dists, axis=1).values  # (N,)
    return dists


def trajectoryOnRoad(trajectory, mapInView):
    viewLength = 27 # mapInView view scope 
    arraySubcornerLength = viewLength / float(39-1) # distance between two pixels 
    mapInView = torch.flip(mapInView, [0]).to(device)  
    unNormedTrajectory = torch.where(trajectory != 0, trajectory * norm.view(60), torch.zeros_like(trajectory))
    
    # converting vectorised trajectory to absolute points
    trajectory = convertRelativeDxDyTensorAbsolutePointListTorch(unNormedTrajectory) 
    
    # length of trajectory inside the mapInView
    lengthInsideMap = lengthInsideMapInView(trajectory) 
    
    # get set of points in the mapInView feature
    roadPoints = torch.where(mapInView == 1) 
    x,y = roadPoints[0], roadPoints[1]  
    
    # converting pixel to mapInView system
    points = torch.stack([x * arraySubcornerLength,viewLength/2 - y * arraySubcornerLength], axis=1) 
    trajectory = trajectory.view(30,2)[:lengthInsideMap]
    
    # calculating chamfer distance between every trajectory point and nearest mapInView point
    dists = chamfLoss(trajectory, points)  
    dists = torch.where(dists < 0.6, dists * 0.0001, dists) 
    dist = torch.sum(dists)/lengthInsideMap
    return dist, lengthInsideMap




