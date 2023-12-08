import os
import matplotlib.pyplot as plt
import matplotlib
from commonroad.visualization.mp_renderer import MPRenderer
import datetime
import moviepy.video.io.ImageSequenceClip
from configparser import ConfigParser
import numpy as np

import commonroad_prediction.HelperFunctions as HelperFunctions
import commonroad_prediction.visualization as visualization
import commonroad_prediction.toPointListConversion as toPointListConversion

''' configuration from config.ini file: '''
file = 'commonroad_prediction/config.ini'  
config = ConfigParser()
config.read(file)

arrayShape = eval(config['var']['arrayShape'])
oneDimension = eval(config['var']['oneDimension'])
viewLength = eval(config['var']['viewLength'])
arraySubcornerLength = eval(config['var']['arraySubcornerLength'])
toleranceOfDetection = eval(config['var']['toleranceOfDetection'])
masterTimesteps = eval(config['var']['masterTimesteps'])
velocityArrowScale = eval(config['var']['velocityArrowScale'])
cutOffTimeSteps = eval(config['var']['cutOffTimeSteps'])
buildImagesAndVideo = eval(config['var']['buildImagesAndVideo'])
egoVehicleCrID = eval(config['var']['egoVehicleCrID'])
fps = eval(config['var']['fps'])
predicionVisible = eval(config['var']['predicionVisible'])


def create_fig_and_ax_for_visualistion(time_begin, time_end, scenario):
    fig, ax = plt.subplots(figsize=None)

    rnd = MPRenderer()
    rnd.draw_params.time_begin = time_begin
    rnd.draw_params.time_end = time_end
    scenario.draw(rnd)
    rnd.render()

    if(time_begin == 0):
        global xmin, xmax, ymin, ymax
        xmin, xmax = ax.get_xlim() # ax.set_xlim ensures that the x-axis is not rescaled when plotting the predictions # ToDo check if this should be fixed as global variable
        ymin, ymax = ax.get_ylim() 

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    return fig, ax

def draw_ground_truth_in_visualisation(ground_truth_dict, time_begin, ax):
    values_by_key = {} # create a dictionary to store the x and y values for each key

    for t in ground_truth_dict:
        if t < time_begin: # skip times before time_begin
            continue
        if t > time_begin + 30: # only display 30 steps of ground_truth to not overload up the visualisation
            break
        for key in ground_truth_dict[t]: # loop over the inner dictionary
            position = ground_truth_dict[t][key]['position']
            x = position[0]
            y = position[1]
            if key not in values_by_key:
                values_by_key[key] = {'x': [], 'y': []}
            values_by_key[key]['x'].append(x)
            values_by_key[key]['y'].append(y)
    ax.plot([], [], color='#0065bd', linestyle='--', linewidth=0.8, # TUMBlue color
            label='Ground Truth')  # add a dummy line for the legend
    for key, values in values_by_key.items():
        ax.plot(values['x'], values['y'], color='#0065bd', # TUMBlue color
                linestyle='--', linewidth=1.0, zorder=30) # plot a separate line for each key linewidth=0.8
    ax.legend(loc='upper left', fontsize=5)  # add the legend to the plot
    return ax

def draw_ID_into_ax(dict, cr_id, ax):
    # draws IDs of all cars of scenario into ax object
    xy = dict[cr_id]['position']
    x, y = xy[0], xy[1]
    id = f'ID: {str(cr_id)}'
    ax.text(x, y, id, fontsize=4, va='center', ha='left', zorder=100)
    return ax


def draw_pictogram_of_sourunding_vehicles_orientation_and_velocity(fig, mapInView, cr_id, groundTruthTrajectoryInView, socialInformationOfCarsInView):

    cmap = matplotlib.colormaps['Greys']
    norm = plt.Normalize(0, 1)
    # ax0 = fig.add_subplot(1, 1, 1)
    ax0 = fig.add_subplot(5, 5, 5)
    textstr = 'cr_ID: '+str(cr_id)
    ax0.set_ylabel(textstr, fontsize=8)
    # ax0 = draw_velocity_arrows_in_plt(ax0, socialInformationOfCarsInView) # Draw the arrows(representing velocites of other cars) in the pictogram
    plt.xticks([int(arrayShape[0]) // 2], ['___\n( "o)\n(o\./o)\n"   "'], color='red')
    # plt.xticks([int(arrayShape[0]) // 2], ['___\n( "o)\n(o\./o)\n"   "\nEgo Vehicle"'], color='red')
    plt.yticks([arrayShape[0]-1,0],  ['0m', '27m'])
    im = ax0.imshow(mapInView , cmap=cmap, zorder=100, norm=norm, alpha=1.0)
    
    # cmap = matplotlib.colormaps['RdBu']
    # norm = plt.Normalize(0, 1)
    # ax2= fig.add_subplot(5, 5, 15)
    # im2 = ax2.imshow(groundTruthArray , cmap=cmap, zorder=100, norm=norm) # old version groundTruthArray
    # cbar = plt.colorbar(im2)
    # cbar.set_label('GroundTruth')

    # cmap = matplotlib.colormaps['Reds'] # ToDo: check if this is good visualisation
    # norm = plt.Normalize(0, 1)
    # # ax3= fig.add_subplot(5, 5, 20)
    # ax3= fig.add_subplot(3, 3, 9)
    # im3 = ax3.imshow(groundTruthTrajectoryInView , cmap=cmap, zorder=100, norm=norm) # new version groundTruthArrayDxDy
    # plt.xticks([int(arrayShape[0])//2], ['___\n( "o)\n(o\./o)\n"   "\nEgo Vehicle'])
    # plt.yticks([arrayShape[0]-1,0],  ['0m', '27m'])
    # cbar = plt.colorbar(im3) # ToDo: check if this is good visualisation
    # cbar.set_label('groundTruthTrajectoryInView') # ToDo: check if this is good visualisation
    return fig

def draw_velocity_arrows_in_plt(ax, socialInteractionsBetweenVehicles):
    arrow_props = dict(facecolor='yellow',linewidth=0.13,alpha = 0.99) # arrow style
    # arrow_props = dict(facecolor='white',fill = True,alpha = 1, edgecolor='blue' , arrowstyle='->', linewidth=0.7) # arrow style
    for i in range(socialInteractionsBetweenVehicles.shape[0]):
        # 1. Get values
        x = socialInteractionsBetweenVehicles[i][0]
        y = socialInteractionsBetweenVehicles[i][1]
        dx = socialInteractionsBetweenVehicles[i][2]
        dy = socialInteractionsBetweenVehicles[i][3]
        # print('x ', x, ' y ', y, ' dx ', dx, ' dy ', dy)
        if x < 0 or y < -viewLength/2.0 or y > viewLength/2.0: # drop if out of view 
            continue
        # 2. Transform to picureCoordinates
        xPrime = - y # xPrime = viewLength/2.0 - y
        yPrime = x
        dxPrime = -dy
        dyPrime = dx

        # 3. Normalize
        xNormd = HelperFunctions.normalize_value(xPrime, -viewLength/2, viewLength/2)
        yNormd = HelperFunctions.normalize_value(yPrime, 0, viewLength)
        dxNormd = HelperFunctions.normalize_value(dxPrime, 0, velocityArrowScale) 
        dyNormd = HelperFunctions.normalize_value(dyPrime, 0, velocityArrowScale) 

        # 4. Draw 
        if(dx == 0 and dy == 0 and x > 0): # drop if velocity is zero
            ax.annotate('', xy=(xNormd, yNormd+0.04), xycoords='axes fraction', xytext=(xNormd, yNormd), textcoords='axes fraction', arrowprops=arrow_props, zorder=500)
            ax.annotate('', xy=(xNormd-0.04, yNormd), xycoords='axes fraction', xytext=(xNormd, yNormd), textcoords='axes fraction', arrowprops=arrow_props, zorder=500)
            ax.annotate('', xy=(xNormd+0.04, yNormd), xycoords='axes fraction', xytext=(xNormd, yNormd), textcoords='axes fraction', arrowprops=arrow_props, zorder=500)
            ax.annotate('', xy=(xNormd, yNormd-0.04), xycoords='axes fraction', xytext=(xNormd, yNormd), textcoords='axes fraction', arrowprops=arrow_props, zorder=500)
            ax.annotate('', xy=(xNormd+dxNormd, yNormd+dyNormd), xycoords='axes fraction', xytext=(xNormd, yNormd),
                textcoords='axes fraction', arrowprops=arrow_props, zorder=500)
    
    ax.plot([], [], color='yellow', label='road user', linewidth=2) # Add a legend for the ground truth and predicted trajectories
    
    return ax

def initialise_prediction_image_folder():
    """
    Creates a folder named `video_trajectory_output` in the current directory to store
    the PNG image frames generated by the `create_prediction_images` function.

    Returns:
    - image_folder (str): The path of the folder created to store the PNG image frames.
    """
    image_folder = 'video_trajectory_output'
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    return image_folder

def make_trajectory_video(image_folder, SCENARIO_PATH):
    """
    Creates a video clip from a sequence of PNG images located in `image_folder` with `masterTimestep` frames,
    and saves the output video with a name of the format 'trajectory_prediction_{filename}.mp4'.
    The video clip is created using `fps` frames per second.

    Optional: Deletes the input PNG files to save disk space.

    Args:
    - image_folder (str): The path of the folder that contains the input image frames.
    - scenario_path (str): The path of the scenario file used to generate the input image frames.


    Returns: None
    """
    image_files = []
    for img_number in range(0, masterTimesteps-cutOffTimeSteps): # ToDo: cutOffTimeSteps should mabye not be used here
        image_files.append(image_folder + '/frame_' + str(img_number) + '.png')

    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(SCENARIO_PATH))[0]
    # time and date information
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H:%M:%S")

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip( image_files, fps)
    clip.write_videofile( image_folder + '/trajectory_prediction_' + filename + "__" + timestamp + '.mp4')

    # (Optional) Delete the PNG files to save disk space
    for file_name in os.listdir(image_folder+'/'):
        if file_name.endswith('.png'):
            os.remove(os.path.join(image_folder+'/', file_name))


def hightlight_car_in_plt_and_visualise_pictogram(dict, ax, fig, mapInView, cr_id, groundTruthTrajectoryInView, socialInformationOfCarsInView):
    x0, y0 = dict[cr_id]['position'][0], dict[cr_id]['position'][1]
    circle = plt.Circle((x0, y0), 6.0, color='red', fill=False, zorder=300)
    ax.add_artist(circle)
    fig = draw_pictogram_of_sourunding_vehicles_orientation_and_velocity(fig, mapInView, cr_id, groundTruthTrajectoryInView, socialInformationOfCarsInView)
    return fig, ax

def visualisation_scenario_prediction_one_timestep(time_begin, time_end, scenario, predictions, ground_truth_dict, SCENARIONAME, adeList, fdeList, comprehensiveTrainingData):
    fig, ax = create_fig_and_ax_for_visualistion(time_begin, time_end, scenario) # add car ids in video
    for cr_id in ground_truth_dict[time_begin]:
        ax = draw_ID_into_ax(ground_truth_dict[time_begin], cr_id, ax) # ToDo: draw all car IDs into ax 
        if cr_id == egoVehicleCrID and buildImagesAndVideo: ####ToDo impleent this later####
            mapInView = comprehensiveTrainingData[cr_id][time_begin]['mapInView']
            groundTruthTrajectoryInView = comprehensiveTrainingData[cr_id][time_begin]['groundTruthTrajectoryInView']
            socialInformationOfCarsInView = comprehensiveTrainingData[cr_id][time_begin]['socialInformationOfCarsInView']
            fig, ax = hightlight_car_in_plt_and_visualise_pictogram(ground_truth_dict[time_begin], ax, fig, mapInView, cr_id, groundTruthTrajectoryInView, socialInformationOfCarsInView)
    
    ax = draw_ground_truth_in_visualisation(ground_truth_dict, time_begin, ax) # draw ground truth trajecoties in output
    if (predicionVisible == True):
        visualization.draw_uncertain_predictions(predictions, ax) # draw predictions in output

    name = os.path.basename(SCENARIONAME)  # for first 10 scenarios of sumo
    ax.set_aspect('equal')  # for crooswalk pedestrian
    title = ax.set_title(f'Scenario: {name} \n')
    title.set_fontsize(10)  # Adjust the font size as needed

    plt.subplots_adjust(bottom=0.125)  # Increase bottom margin to make space for the longer label
    secax = ax.secondary_xaxis('bottom')

    secax.set_xlabel("ADE: {:.3f}     meanADE: {:.3f}\nFDE: {:.3f}     meanFDE: {:.3f}".format(adeList[-1], np.mean(adeList), fdeList[-1], np.mean(fdeList)))
    return fig


def drawPictogramOfmapInView(mapInView,egoVelocity, ax):
    # print(mergedArray)
    # print(f'\n\negoVelocity = {egoVelocity}')
    flipped_mapInView = np.flipud(mapInView)
    # fig = plt.figure()
    cmap = plt.get_cmap('Greys')
    norm = plt.Normalize(0, 1)
    ax.imshow(flipped_mapInView, cmap=cmap, norm=norm)
    return ax

def plotOutputVectors(outputPointList, groundTruthTrajectoryPointList, ax, loss):
    # outputPointList = outputPointList.view(30,2)
    # groundTruthTrajectoryPointList = groundTruthTrajectoryPointList.view(30,2)
    outputPointList = toPointListConversion.convertRelativeDxDyTensorAbsolutePointList(outputPointList).view(30,2)
    groundTruthTrajectoryPointList = toPointListConversion.convertRelativeDxDyTensorAbsolutePointList(groundTruthTrajectoryPointList).view(30,2)
    print(f'groundTruthTrajectoryPointList = {groundTruthTrajectoryPointList}')
    rotation_matrix = np.array([[0, -1], [1, 0]]) # Rotate the vectors counterclockwise by 90 degrees
    outputPointList = np.dot(outputPointList, rotation_matrix.T)
    groundTruthTrajectoryPointList = np.dot(groundTruthTrajectoryPointList, rotation_matrix.T)

    metaFactor = 20/27
    middle = 9
    
    arrow_props05s = dict(facecolor='red', fill=True, alpha=1, edgecolor='white', arrowstyle='->', linewidth=2)  # arrow style
    
    # Plot each x and y pair on the axis
    x_last, y_last = middle, 0
    arrow_props = dict(facecolor='red', fill=True, alpha=1, edgecolor='red', arrowstyle='->', linewidth=2)  # arrow style
    for i in range(len(outputPointList)):
        if i > 0: x_last, y_last = outputPointList[i-1][0]*metaFactor + middle, outputPointList[i-1][1]*metaFactor 
        x, y =outputPointList[i][0]*metaFactor + middle, outputPointList[i][1]*metaFactor
        ax.annotate('', xy=(x, y), xytext=(x_last, y_last), arrowprops=arrow_props)
        if(i%5 == 0): ax.annotate('', xy=(x, y), xytext=(x_last, y_last), arrowprops=arrow_props05s)
    # Plot each x and y pair on the axis
    x_last, y_last = middle, 0
    arrow_props = dict(facecolor='#0065bd', fill=True, alpha=1, edgecolor='#0065bd', arrowstyle='->', linewidth=2)  # arrow style
    for i in range(len(groundTruthTrajectoryPointList)):
        if i > 0: x_last, y_last = groundTruthTrajectoryPointList[i-1][0]*metaFactor+  middle, groundTruthTrajectoryPointList[i-1][1]*metaFactor
        x, y =groundTruthTrajectoryPointList[i][0]*metaFactor+ middle, groundTruthTrajectoryPointList[i][1]*metaFactor
        ax.annotate('', xy=(x, y), xytext=(x_last, y_last), arrowprops=arrow_props)
        if(i%5 == 0): ax.annotate('', xy=(x, y), xytext=(x_last, y_last), arrowprops=arrow_props05s)
    


    ax.set_ylim([0, 18])  # Set x-axis limits
    ax.set_xlim([0, 18])  # Set y-axis limits
    ax.set_xticks([middle])  # Set the x-axis tick positions
    ax.set_xticklabels(['___\n( "o)\n(o\./o)\n"   "\nEgo Vehicle'])  # Set the x-axis tick labels
    ax.set_yticks([18,0])  # Set the y-axis tick positions
    ax.set_yticklabels(['27m', '0m'])  # Set the y-axis tick labels
    ax.plot(middle, 18, marker='o', markersize=5, color='green')
    ax.text(1, -3, f"Loss: {loss:.5f}", color='red', ha='center', va='center', fontsize=12)

    # # plt.show()
    # return plt
    
def plotOutputVectorsDxDy(outputPointList, groundTruthTrajectoryPointList, ax, loss):
    outputPointList = toPointListConversion.convertRelativeDxDyTensorAbsolutePointList(outputPointList)
    groundTruthTrajectoryPointList = toPointListConversion.convertRelativeDxDyTensorAbsolutePointList(groundTruthTrajectoryPointList)
    rotation_matrix = np.array([[0, -1], [1, 0]]) # Rotate the vectors counterclockwise by 90 degrees
    outputPointList = np.dot(outputPointList, rotation_matrix.T)
    groundTruthTrajectoryPointList = np.dot(groundTruthTrajectoryPointList, rotation_matrix.T)

    metaFactor = 20/27
    middle = 9
    
    arrow_props05s = dict(facecolor='red', fill=True, alpha=1, edgecolor='white', arrowstyle='->', linewidth=2)  # arrow style
    
    # Plot each x and y pair on the axis
    x_last, y_last = middle, 0
    arrow_props = dict(facecolor='blue', fill=True, alpha=1, edgecolor='blue', arrowstyle='->', linewidth=2)  # arrow style
    for i in range(len(groundTruthTrajectoryPointList)):
        if i > 0: x_last, y_last = groundTruthTrajectoryPointList[i-1][0]*metaFactor+  middle, groundTruthTrajectoryPointList[i-1][1]*metaFactor
        x, y =groundTruthTrajectoryPointList[i][0]*metaFactor+ middle, groundTruthTrajectoryPointList[i][1]*metaFactor
        ax.annotate('', xy=(x, y), xytext=(x_last, y_last), arrowprops=arrow_props)
        if(i%5 == 0): ax.annotate('', xy=(x, y), xytext=(x_last, y_last), arrowprops=arrow_props05s)
    
    # Plot each x and y pair on the axis
    x_last, y_last = middle, 0
    arrow_props = dict(facecolor='red', fill=True, alpha=1, edgecolor='red', arrowstyle='->', linewidth=2)  # arrow style
    for i in range(len(outputPointList)):
        if i > 0: x_last, y_last = outputPointList[i-1][0]*metaFactor + middle, outputPointList[i-1][1]*metaFactor 
        x, y =outputPointList[i][0]*metaFactor + middle, outputPointList[i][1]*metaFactor
        ax.annotate('', xy=(x, y), xytext=(x_last, y_last), arrowprops=arrow_props)
        if(i%5 == 0): ax.annotate('', xy=(x, y), xytext=(x_last, y_last), arrowprops=arrow_props05s)


    ax.set_ylim([0, 18])  # Set x-axis limits
    ax.set_xlim([0, 18])  # Set y-axis limits
    ax.set_xticks([middle])  # Set the x-axis tick positions
    ax.set_xticklabels(['___\n( "o)\n(o\./o)\n"   "\nEgo Vehicle'])  # Set the x-axis tick labels
    ax.set_yticks([18,0])  # Set the y-axis tick positions
    ax.set_yticklabels(['27m', '0m'])  # Set the y-axis tick labels
    ax.plot(middle, 18, marker='o', markersize=5, color='green')
    ax.text(1, -3, f"Loss: {loss:.5f}", color='red', ha='center', va='center', fontsize=12)

    # # plt.show()
    # return plt