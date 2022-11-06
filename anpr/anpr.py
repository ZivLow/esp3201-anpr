# Github Repository: https://github.com/ZivLow/esp3201-anpr
# Script for performing automatic number plate recognition 
# and identifying number plates of speeding vehicles
# ESP3201 (Machine Learning in Robotics and Engineering) Project
# 1) Ronald Wee
# 2) Ziv Low
# 
# Tested and run on Ubuntu 22.04 LTS
# 
# USAGE: Install all needed dependencies. 
#        Recommended to use the provided shell scripts 'run_anpr.sh' or 'custom_run_anpr.sh' in 'esp3201-anpr' directory
#        Run in bash terminal: './run_anpr.sh' or './custom_run_anpr.sh' in 'esp3201-anpr' directory
#        You may need to enable execute permissions for these shell scripts by running 'chmod 775 run_anpr.sh' or 'chmod 775 custom_run_anpr.sh'
# 
#        Alternatively, you can also run 'python anpr.py' inside 'anpr' directory.

import math
import os
import cv2
import dlib
import easyocr
import pytesseract
import torch
import argparse
import json
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import ExifTags, Image, ImageOps
from time import process_time as get_time

# Install ALPR-SDK properly to import 'ultimateAlprSdk' 
# to be able to use ALPR_SDK as a vehicle, plate, or OCR detection method
# They provided a sample python file to allow you to check if install was successful or not.
# See https://github.com/DoubangoTelecom/ultimateALPR-SDK
try:
    import ultimateAlprSdk
except ModuleNotFoundError as err:
    print(err)

# Context manager to time a block of code
class time_this_code(object):

    # Entering the context manager
    def __enter__(self):
        self.time = get_time()
        return self
    
    # Exiting the context manager
    def __exit__(self, type, value, traceback):
        self.time = round((get_time() - self.time) * 1000)

# Decorator get the process time of the function. The time is assigned to the global variable function_duration
def time_this_function(f):
    def timer(*args, **kwargs):
        global function_duration

        # Begin timing
        start_time = get_time()
        
        # Run the function f
        result = f(*args, **kwargs)

        # End timing
        end_time = get_time()

        # Time duration
        function_duration = round((end_time - start_time) * 1000)

        return result
    return timer

# Utility function to draw dotted line for opencv
def drawline(img,pt1,pt2,color,thickness=1.0,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

# Utility function to draw dotted polygon for opencv
def drawpoly(img,pts,color,thickness=1.0,style='dotted'):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

# Utility function to draw dotted rectangle for opencv
def drawrect(img,pt1,pt2,color,thickness=1.0,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style)

# Load image from opencv video frame
def load_pil_image_from_opencv_frame(opencv_image):

    # convert from openCV2 to PIL.
    color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_converted)

    # Get image exif data
    img_exif = pil_image.getexif()
    ret = {}
    orientation  = 1
    try:
        if img_exif:
            for tag, value in img_exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                ret[decoded] = value
            orientation  = ret["Orientation"]
    except Exception as e:
        print("An exception occurred: {}".format(e))
        traceback.print_exc()

    if orientation > 1:
        pil_image = ImageOps.exif_transpose(pil_image)

    if 'IMAGE_TYPES_MAPPING' in globals():
        if pil_image.mode in IMAGE_TYPES_MAPPING:
            image_type = IMAGE_TYPES_MAPPING[pil_image.mode]
        else:
            raise ValueError(TAG + "Invalid mode: %s" % pil_image.mode)
    else:
        image_type = None

    return pil_image, image_type

# Function to convert bounding box (x_min, y_min, x_max, y_max) to (x_min, y_min, width, height)
def convert_xyxy_to_xywh(_x_min, _y_min, _x_max, _y_max):

    _width = _x_max - _x_min
    _height = _y_max - _y_min

    return tuple(map(int, (_x_min, _y_min, _width, _height)))

# Function to get center position (x, y) given bounding box (x_min, y_min, x_max, y_max)
def get_center_position(_bounding_box):
    _x_center = 0.5 * (_bounding_box[0] + _bounding_box[2])
    _y_center = 0.5 * (_bounding_box[1] + _bounding_box[3])
    return _x_center, _y_center

def init_alpr_sdk():

    global TAG
    global IMAGE_TYPES_MAPPING

    TAG = "[PythonRecognizer] "

    IMAGE_TYPES_MAPPING = { 
            'RGB': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGB24,
            'RGBA': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGBA32,
            'L': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_Y
    }

    # Defines the default JSON configuration. More information at https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html
    JSON_CONFIG = {
        "debug_level": "warn",
        "debug_write_input_image_enabled": False,
        "debug_internal_data_path": ".",
        
        "num_threads": -1,
        "gpgpu_enabled": True,
        "max_latency": -1,

        "klass_vcr_gamma": 1.5,
        
        "detect_roi": [0, 0, 0, 0],
        "detect_minscore": 0.1,

        "car_noplate_detect_min_score": 0.8,
        
        "pyramidal_search_enabled": True,
        "pyramidal_search_sensitivity": 0.28,
        "pyramidal_search_minscore": 0.3,
        "pyramidal_search_min_image_size_inpixels": 800,
        
        "recogn_rectify_enabled": True,
        "recogn_minscore": 0.3,
        "recogn_score_type": "min"
    }

    # Update JSON options using values from the command args
    JSON_CONFIG["assets_folder"] = alpr_sdk_asset_path
    JSON_CONFIG["charset"] = "latin"
    JSON_CONFIG["car_noplate_detect_enabled"] = True
    JSON_CONFIG["ienv_enabled"] = False
    JSON_CONFIG["openvino_enabled"] = False
    JSON_CONFIG["openvino_device"] = "CPU"
    JSON_CONFIG["npu_enabled"] = True
    JSON_CONFIG["klass_lpci_enabled"] = False
    JSON_CONFIG["klass_vcr_enabled"] = False
    JSON_CONFIG["klass_vmmr_enabled"] = False
    JSON_CONFIG["klass_vbsr_enabled"] = False
    JSON_CONFIG["license_token_file"] = ""
    JSON_CONFIG["license_token_data"] = ""

    # Initialize the ALPR SDK engine
    checkResult("Init", ultimateAlprSdk.UltAlprSdkEngine_init(json.dumps(JSON_CONFIG)))

# Check result
def checkResult(operation, result):
    if not result.isOK():
        print(TAG + operation + ": failed -> " + result.phrase())
        assert False
    else:
        print(TAG + operation + ": OK -> " + result.json())

# Function to perform OCR using EasyOCR
def easyocr_text_extraction(input_image, plate_box):

    # Plate region of interest
    roi = input_image[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]

    # Perform OCR
    ocr_results = easyocrReader.readtext(image=roi, decoder='greedy', add_margin=0.1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', paragraph=False)
    
    # If no ocr_results
    if not ocr_results:
        return None, None

    plate_number = ''
    ocr_confidence = 0.0

    # Loop through ocr_results
    for ocr_result in ocr_results:
        plate_number = plate_number + ocr_result[1]
        ocr_confidence += ocr_result[2]
    
    ocr_confidence = round(ocr_confidence*100/len(ocr_results), 1)

    # Return plate number, and ocr confidence
    return plate_number, ocr_confidence

# Function to perform OCR using PyTesseract
def tesseract_text_extraction(input_image, plate_box):

    # pytesseract_config
    pytesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"'

    # Plate region of interest
    roi = input_image[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]

    # Multiple ocr confidences
    ocr_confidences = pytesseract.image_to_data(roi, config=pytesseract_config, output_type=pytesseract.Output.DICT)['conf']

    # Get valid ocr_confidences. Filter out nonsense ocr results.
    ocr_confidences = [x for x in ocr_confidences if x != -1]

    # If filtered ocr results is not empty
    if ocr_confidences:
        # Compute average ocr_confidence amongst the valid results
        ocr_confidence = round(sum(ocr_confidences) / len(ocr_confidences), 1)

        # Extracted text
        predicted_result = pytesseract.image_to_string(roi, lang='eng', config=pytesseract_config)

        # Join the text
        plate_number = "".join(predicted_result.split()).replace(":", "").replace("-", "")
    
    # If filtered ocr results is empty:
    else:
        plate_number = None
        ocr_confidence = None

    return plate_number, ocr_confidence

# Function to perform OCR using ALPR-SDK
def alpr_sdk_text_extraction(input_image, plate_box):

    image, imageType = load_pil_image_from_opencv_frame(input_image)
    width, height = pil_image.size

    # Get the result from ALPR-SDK
    alpr_output_result = ultimateAlprSdk.UltAlprSdkEngine_process(
            imageType,
            image.tobytes(), # type(x) == bytes
            width,
            height,
            0, # stride
            1 # exifOrientation (already rotated in load_image -> use default value: 1)
        )

    # Get the json output from ALPR as dictionary
    alpr_output = json.loads(alpr_output_result.json())

    # Number of plates detected
    num_plates_detected = alpr_output_result.numPlates()

    plate_number = None
    ocr_confidence = None

    # If plates are detected
    if num_plates_detected > 0:

        current_plate = alpr_output['plates'][0]

        # Slice into plate_number
        plate_number = current_plate.get('text', None)

        # Slice into ocr_confidence
        if current_plate.get('confidences', None) is not None:
            ocr_confidence = round(current_plate['confidences'][0], 1)

    return plate_number, ocr_confidence



# Function to perform OCR.
@time_this_function
def perform_ocr(input_image, plate_box, ocr_method='EasyOCR'):
    match ocr_method:
        case 'EasyOCR':
            return easyocr_text_extraction(input_image, plate_box)
        
        case 'PyTesseract':
            return tesseract_text_extraction(input_image, plate_box)

        case 'ALPR_SDK':
            return alpr_sdk_text_extraction(input_image, plate_box)
        
        case _:
            raise ValueError("OCR method argument must be 'EasyOCR' or 'PyTesseract'.")

# Function to resize image
def resize2SquareKeepingAspectRation(img, size, interpolation):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv2.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv2.resize(mask, (size, size), interpolation)

# Function to clear out boxes which have poor tracking quality
def remove_poor_trackers(input_image, _attributes, key='tracker', threshold=7):

    IDtoDelete = []

    # Loop through currently being tracked objects
    for ID in _attributes.keys():
        trackerQuality = _attributes[ID][key].update(input_image)
        
        # If poor tracking quality or the vehicle has been moving very slowly for quite some time
        if (trackerQuality < threshold) or (_attributes[ID].get('slow_vehicle_counter', 0) * TRACKER_FRAME_CHECK_INTERVAL / input_video_fps  > SLOW_MOVING_TIME_THRESHOLD):
            IDtoDelete.append(ID)
    
    # Delete those with poor tracking quality
    for ID in IDtoDelete:
        if TRACKER_DEBUG: print (f'Removing ID {ID} from list of {key}.')
        _attributes.pop(ID, None)

    return _attributes

# Check if new bounding box is within an old bounding box
def isNewBox(_bounding_box, _attributes, key='tracker'):

    # Get corner points
    x_min, y_min, x_max, y_max = _bounding_box

    # Average center position (x_center, y_center)
    x_center, y_center = get_center_position(_bounding_box)

    matchID = None

    # Loop through all vehicles being tracked
    for ID in _attributes.keys():
        trackedPosition = _attributes[ID][key].get_position()
        
        t_x = int(trackedPosition.left())
        t_y = int(trackedPosition.top())
        t_w = int(trackedPosition.width())
        t_h = int(trackedPosition.height())
        
        t_x_bar = t_x + 0.5 * t_w
        t_y_bar = t_y + 0.5 * t_h

        # Check if the new bounding box is within already tracked bounding box
        if (((t_x <= x_center <= (t_x + t_w)) and (t_y <= y_center <= (t_y + t_h))) or ((x_min <= t_x_bar <= x_max) and (y_min <= t_y_bar <= y_max))):
            matchID = ID

            break

    return matchID

# Function to track all vehicles
def trackAllVehicles(_image, _attributes, colour=(0, 255, 0)):

    # Loop through vehicles being tracked
    for vehicleID in _attributes.keys():

        # Tracked position of vehicle
        vehicleTrackedPosition = _attributes[vehicleID]['vehicle_tracker'].get_position()
        t_x = int(vehicleTrackedPosition.left())
        t_y = int(vehicleTrackedPosition.top())
        t_w = int(vehicleTrackedPosition.width())
        t_h = int(vehicleTrackedPosition.height())

        # Draw bounding box for vehicle detected
        cv2.rectangle(_image, (t_x, t_y), (t_x + t_w, t_y + t_h), colour, 2)

        # Write the confidence for the vehicle
        vehicle_confidence = _attributes[vehicleID].get('vehicle_confidence', None)

        if vehicle_confidence is None:
            cv2.putText(_image, " Vehicle", (t_x, t_y - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)
        else:
            vehicle_confidence_string = str(round(vehicle_confidence, 1))
            cv2.putText(_image, " Vehicle (" + vehicle_confidence_string + "%)", (t_x, t_y - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)

        if (_attributes[vehicleID].get('speed_captured', False) == True):
            cv2.putText(_image, str(_attributes[vehicleID].get('speed')) + " km/hr", (t_x, t_y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 1)

        # If specified number of frames has passed
        if frameCounter % TRACKER_FRAME_CHECK_INTERVAL == 0:

            # For speed estimation
            _attributes[vehicleID]['location2'] = (t_x, t_y, t_w, t_h)

    return _image, _attributes

# Function to track all plates
def trackAllPlates(_image, _attributes, colour=(200, 0, 255)):

    # Loop through plates being tracked
    for plateID in _attributes.keys():

        # Tracked position of vehicle
        plateTrackedPosition = _attributes[plateID]['plate_tracker'].get_position()
        t_x = int(plateTrackedPosition.left())
        t_y = int(plateTrackedPosition.top())
        t_w = int(plateTrackedPosition.width())
        t_h = int(plateTrackedPosition.height())

        # Draw bounding box for plate
        cv2.rectangle(_image, (t_x, t_y), (t_x + t_w, t_y + t_h), colour, 2)

        # Write the confidence for the plate
        plate_confidence_string = str(round(_attributes[plateID]['plate_confidence'], 1))
        cv2.putText(_image, " Plate (" + plate_confidence_string + "%)", (t_x, t_y - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)

        # If OCR managed to get plate_number
        if _attributes[plateID]['plate_number']:
            # Write the plate number
            cv2.putText(_image, str(_attributes[plateID]['plate_number']) + " (" + str(_attributes[plateID]['ocr_confidence']) + "%)", (t_x, t_y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)
        
    return _image, _attributes

# Function to get speed of tracked vehicles
def get_vehicle_speed(_image, _attributes, _frameCounter, boundary=(100, 400, 900, 900), colour=(0, 255, 0)):

    # Compute speed
    for vehicleID in _attributes.keys():

        # Check speed after specified number of frames has passed
        if _frameCounter % TRACKER_FRAME_CHECK_INTERVAL == 0:

            # Old vehicle location
            x1, y1, w1, h1 = _attributes[vehicleID]['location1']

            # New vehicle location computed by tracker
            x2, y2, w2, h2 = _attributes[vehicleID]['location2']
            
            # Update old vehicle location with new tracked location
            _attributes[vehicleID]['location1'] = (x2, y2, w2, h2)

            # If the vehicle has moved
            if [x1, y1, w1, h1] != [x2, y2, w2, h2]:

                # Get threshold boundary
                thres_x_min, thres_y_min, thres_x_max, thres_y_max = boundary

                old_center = get_center_position((x1, y1, x1 + w1, y1 + h1))
                new_center = get_center_position((x2, y2, x2 + w2, y2 + h2))

                # Get previous speed
                old_speed = _attributes[vehicleID].get('speed', None)

                # Compute speed
                estimated_speed = estimateSpeed(old_center, new_center, input_video_fps)

                # If within speed capture box
                if (old_speed == None or estimated_speed > old_speed) and (thres_x_min <= new_center[0] <= thres_x_max) and (thres_y_min <= new_center[1] <= thres_y_max):
                    _attributes[vehicleID]['speed'] = estimated_speed
                    _attributes[vehicleID]['speed_captured'] = True
                
                # Check if tracked object is moving very slowly
                elif estimated_speed < SLOW_MOVING_SPEED_THRESHOLD:
                    if _attributes[vehicleID].get('slow_vehicle_counter', None) is None:
                        _attributes[vehicleID]['slow_vehicle_counter'] = 0
                    _attributes[vehicleID]['slow_vehicle_counter'] += 1


    return _image, _attributes

# Function to estimate speed using ppm
def estimateSpeed(location1, location2, frame_rate):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # PPM = location2[2] / carWidth
    d_meters = d_pixels / PPM
    #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    #speed = d_meters * frame_rate * 3.6 / TRACKER_FRAME_CHECK_INTERVAL
    speed = d_meters * frame_rate * 3.6 / TRACKER_FRAME_CHECK_INTERVAL
    return round(speed, 1)

# Calculates Intersection over Union (IoU) of bb1 and bb2
# Taken from: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation 
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# Check if _object1_bb (xyxy) is within _object2_bb (xyxy)
def isWithinBB(_object1_bb, _object2_bb, threshold=0.8):

    dict_keys = ('x1', 'y1', 'x2', 'y2')

    # Convert bounding box format: tuple --> dictionary
    _bb1 = {key: value for key, value in zip(dict_keys, _object1_bb)}
    _bb2 = {key: value for key, value in zip(dict_keys, _object2_bb)}

    # Return true if there is intersection
    return True if math.ceil(get_iou(_bb1, _bb2)) else False

# Check through all plates to determine if they are within a vehicle
def match_plate_with_vehicle(_vehicleAttributes, _plateAttributes, threshold=0.8):

    
    vehicles_with_plates = {}

    vehicles = _vehicleAttributes.copy()
    plates = _plateAttributes.copy()

    # Loop through all plates
    for plateID in plates:

        # Skip plate if it does not have plate_number
        if plates[plateID].get('plate_number', None) is None:
            continue
        
        # Plate position
        plate_position = plates[plateID]['plate_tracker'].get_position()
        plate_x_min = plate_position.left()
        plate_y_min = plate_position.top()
        plate_x_max = plate_position.right()
        plate_y_max = plate_position.bottom()
        # Plate bounding box
        plate_bb = (plate_x_min, plate_y_min, plate_x_max, plate_y_max)

        matchedVehicleID = None

        # Loop through all vehicles:
        for vehicleID in vehicles:

            vehicle_speed = vehicles[vehicleID].get('speed', None)

            # Skip vehicle if it does not have speed or is within speed_limit
            if (vehicle_speed is None) or (vehicle_speed <= SPEED_LIMIT):
                continue

            # Vehicle position
            vehicle_position = vehicles[vehicleID]['vehicle_tracker'].get_position()
            vehicle_x_min = vehicle_position.left()
            vehicle_y_min = vehicle_position.top()
            vehicle_x_max = vehicle_position.right()
            vehicle_y_max = vehicle_position.bottom()
            # Vehicle bounding box
            vehicle_bb = (vehicle_x_min, vehicle_y_min, vehicle_x_max, vehicle_y_max)

            # Check if plate is within vehicle, with threshold control
            # This is a greedy approach, as only the first match that fulfils the threshold is considered
            # A less greedy approach would be to not break immediately, but continue looping, then match the pair with largest IoU
            if isWithinBB(plate_bb, vehicle_bb, threshold):

                # Inherit all the attributes from vehicle
                vehicles_with_plates[vehicleID] = vehicles[vehicleID]
                vehicles_with_plates[vehicleID]['vehicle_id'] = vehicleID

                # Add the matched plate attributes
                vehicles_with_plates[vehicleID].update(plates[plateID])
                vehicles_with_plates[vehicleID]['plate_id'] = plateID

                # # Clean up output dictionary
                # for key in ('vehicle_tracker', 'plate_tracker', 'location2', 'speed_captured', 'slow_vehicle_counter'):
                #     vehicles_with_plates[vehicleID].pop(key, None)

                matchedVehicleID = vehicleID

                break
        
        if matchedVehicleID:
            vehicles.pop(matchedVehicleID, None)

    return list(vehicles_with_plates.values())

# Initialize ANPR
def init_anpr(initialize_set):
    if 'haar_cascade' in initialize_set: init_haar_cascade()
    if 'YOLOv5' in initialize_set: init_yolov5()
    if 'ALPR_SDK' in initialize_set: init_alpr_sdk()
    if 'EasyOCR' in initialize_set: init_easyocr()

# Initialize haar_cascade using opencv
def init_haar_cascade():
    global haar_cascade_model
    haar_cascade_model = cv2.CascadeClassifier(haar_cascade_weights_path)

# Initialize YOLOv5 using pyTorch
def init_yolov5():
    global yolov5_model
    yolov5_model = torch.hub.load(yolov5_dir, 'custom', path=yolov5_weights_path, source='local')

# Initialize YOLOv5 using pyTorch
def init_easyocr():
    global easyocrReader
    easyocrReader = easyocr.Reader(['en'], gpu=False)

# Perform inference
@time_this_function
def perform_inference(vehicle_detect_method='YOLOv5', plate_detect_method='YOLOv5'):

    inference_method = [vehicle_detect_method, plate_detect_method]

    if 'haar_cascade' in inference_method:
        haar_cascade_results, rejectLevels, levelWeights = haar_cascade_model.detectMultiScale3(opencv_image, scaleFactor=1.1, minNeighbors=32, minSize=(30, 30), outputRejectLevels = True)
        haar_cascade_vehicles_df = get_haar_cascade_dataframe(haar_cascade_results)

    if 'YOLOv5' in inference_method:
        yolo_inference_results = yolov5_model(opencv_image)
        yolo_results_df = yolo_inference_results.pandas().xyxy[0]
        yolo_results_df['confidence'] = yolo_results_df['confidence'].multiply(100)

        # Get the vehicles
        yolo_vehicles_df = yolo_results_df.loc[yolo_results_df['name'] == 'Land vehicle']

        # Get the plates
        yolo_plates_df = yolo_results_df.loc[yolo_results_df['name'] == 'Vehicle registration plate']

    if 'ALPR_SDK' in inference_method:
        # Get the result from ALPR-SDK
        alpr_sdk_output_result = ultimateAlprSdk.UltAlprSdkEngine_process(
                imageType,
                pil_image.tobytes(), # type(x) == bytes
                width,
                height,
                0, # stride
                1 # exifOrientation (already rotated in load_image -> use default value: 1)
            )
        
        if not alpr_sdk_output_result.isOK():
            err = TAG + "operation" + ": failed -> " + alpr_sdk_output_result.phrase()
            raise ValueError(err)
        
        # Get results in pandas dataframe format
        alpr_sdk_results_df = get_alpr_sdk_dataframe(alpr_sdk_output_result)

        # If items are found
        if len(alpr_sdk_results_df):
            # Get the vehicles
            alpr_sdk_vehicles_df = alpr_sdk_results_df.loc[alpr_sdk_results_df['name'] == 'Land vehicle']

            # Get the plates
            alpr_sdk_plates_df = alpr_sdk_results_df.loc[alpr_sdk_results_df['name'] == 'Vehicle registration plate']

        # If no items are found
        else:
            alpr_sdk_vehicles_df = pd.DataFrame()
            alpr_sdk_plates_df = pd.DataFrame()
        
    # Pattern matching with PEP 636: https://peps.python.org/pep-0636/#or-patterns
    match inference_method:
        case ['haar_cascade', 'YOLOv5']:
            return haar_cascade_vehicles_df, yolo_plates_df

        case ['haar_cascade', 'ALPR_SDK']:
            return haar_cascade_vehicles_df, alpr_sdk_plates_df

        case ['YOLOv5', 'YOLOv5']:
            return yolo_vehicles_df, yolo_plates_df

        case ['YOLOv5', 'ALPR_SDK']:
            return yolo_vehicles_df, alpr_sdk_plates_df

        case ['ALPR_SDK', 'YOLOv5']:
            return alpr_sdk_vehicles_df, yolo_plates_df
        
        case ['ALPR_SDK', 'ALPR_SDK']:
            return alpr_sdk_vehicles_df, alpr_sdk_plates_df

        case _:
            raise ValueError("Invalid 'vehicle_detect_method' or 'plate_detect_method' argument.")

# Convert the haar_cascade_results into usable pandas dataframe
def get_haar_cascade_dataframe(haar_cascade_results):

    haar_cascade_detection_results = []

    # If no detections
    if np.array_equal(haar_cascade_results, ()):
        return pd.DataFrame()

    # Loop through each vehicle boundaries
    for x, y, w, h in haar_cascade_results:
        vehicle_x_min = x
        vehicle_y_min = y
        vehicle_x_max = x + w
        vehicle_y_max = y + h

        new_vehicle = {
            'name': 'Land vehicle',
            'xmin': vehicle_x_min,
            'ymin': vehicle_y_min,
            'xmax': vehicle_x_max,
            'ymax': vehicle_y_max,
            'confidence': None
        }

        haar_cascade_detection_results.append(new_vehicle)

    haar_cascade_detection_results_df = pd.DataFrame.from_records(haar_cascade_detection_results)

    return haar_cascade_detection_results_df


# Convert and filter alpr_sdk json results into usable pandas dataframe
def get_alpr_sdk_dataframe(alpr_sdk_output_result):
    # Get the json output from ALPR as dictionary
    alpr_sdk_output = json.loads(alpr_sdk_output_result.json())

    alpr_sdk_detection_results = []

    # Number of vehicles detected
    num_vehicles_detected = alpr_sdk_output_result.numCars()

    # Number of plates detected
    num_plates_detected = alpr_sdk_output_result.numPlates()

    # If vehicles are detected
    if num_vehicles_detected > 0:

        # Loop through all vehicles found
        for vehicle_idx in range(num_vehicles_detected):

            current_vehicle = alpr_sdk_output['plates'][vehicle_idx].get('car', None)

            # Skip if no value found
            if current_vehicle is None:
                continue

            # Slice into vehicle_box
            vehicle_boundary = list(map(int, current_vehicle['warpedBox']))
            vehicle_x_min = vehicle_boundary[0]
            vehicle_y_min = vehicle_boundary[1]
            vehicle_x_max = vehicle_boundary[4]
            vehicle_y_max = vehicle_boundary[5]

            # Slice into vehicle_confidence
            vehicle_confidence = current_vehicle.get('confidence', None)

            # Skip if no value found
            if vehicle_confidence is None:
                continue

            new_vehicle = {
                'name': 'Land vehicle',
                'xmin': vehicle_x_min,
                'ymin': vehicle_y_min,
                'xmax': vehicle_x_max,
                'ymax': vehicle_y_max,
                'confidence': vehicle_confidence
            }

            alpr_sdk_detection_results.append(new_vehicle)

    # If plates are detected
    if num_plates_detected > 0:

        # Loop through all plates found
        for plate_idx in range(num_plates_detected):

            current_plate = alpr_sdk_output['plates'][plate_idx]

            # Slice into plate_box
            plate_boundary = current_plate.get('warpedBox', None)

            # Skip if no value found
            if plate_boundary is None:
                continue

            plate_x_min = plate_boundary[0]
            plate_y_min = plate_boundary[1]
            plate_x_max = plate_boundary[4]
            plate_y_max = plate_boundary[5]

            # Skip if no value found
            if current_plate.get('confidences', None) is None:
                continue

            # Slice into plate_confidence
            plate_confidence = current_plate['confidences'][1]

            new_plate = {
                'name': 'Vehicle registration plate',
                'xmin': plate_x_min,
                'ymin': plate_y_min,
                'xmax': plate_x_max,
                'ymax': plate_y_max,
                'confidence': plate_confidence
            }

            alpr_sdk_detection_results.append(new_plate)

    alpr_sdk_detection_results_df = pd.DataFrame.from_records(alpr_sdk_detection_results)

    return alpr_sdk_detection_results_df

# Find the arguments supplied to this script
def get_input_args():
    anpr_parser = argparse.ArgumentParser(description="""
    This is a ANPR script for ESP3201 project Group 1.
    """)

    # Processing options
    anpr_parser.add_argument("--vehicle_detect_method", required=False, default='YOLOv5', help="Processing method for vehicle detection. Choose from: 'YOLOv5', 'ALPR_SDK'")
    anpr_parser.add_argument("--plate_detect_method", required=False, default='YOLOv5', help="Processing method for plate detection. Choose from: 'YOLOv5', 'ALPR_SDK'")
    anpr_parser.add_argument("--ocr_detect_method", required=False, default='EasyOCR', help="Processing method for plate alphanumerics. 'EasyOCR', 'PyTesseract', 'ALPR_SDK'")

    # Input video options
    anpr_parser.add_argument("--video_filename", required=False, default='traffic_departing_1', help="filename of input video")
    anpr_parser.add_argument("--video_file_extension", required=False, default='.mov', help="file extension type of input video")
    anpr_parser.add_argument("--video_file_directory", required=False, default='/home/ziv/speeding_catcher/traffic_footage/', help="Directory of input video")
    anpr_parser.add_argument("--video_fps", required=False, default="30", help="video frame rate")

    # Ouput video options
    anpr_parser.add_argument("--output_video", required=False, default="True", help="[True, False]. Whether to output video")
    anpr_parser.add_argument("--output_video_directory", required=False, default='/home/ziv/speeding_catcher/output_video/', help="Directory of output video")
    anpr_parser.add_argument("--output_video_append", required=False, default='_output', help="String to append to output video filename")

    # Output csv file
    anpr_parser.add_argument("--output_csv", required=False, default="True", help="[True, False]. Whether to output csv file")

    # Frame check interval
    anpr_parser.add_argument("--inference_frame_check_interval", required=False, default="1", help="How often to perform inference")
    anpr_parser.add_argument("--tracker_frame_check_interval", required=False, default="1", help="How often to update tracker. Interval of frames to skip, before performing dlib tracking calculations.")

    # Thresholds
    anpr_parser.add_argument("--speed_limit", required=False, default="2", help="+ve float or int. Speed limit of the road in km/hr to track. Minimum speed to consider for matching with plates into csv file")
    anpr_parser.add_argument("--ppm", required=False, default="25.0", help="+ve float or int. Pixels per meter in the speed capture box")
    anpr_parser.add_argument("--vehicle_track_threshold", required=False, default="8", help="[1 ~ 9]. Reject tracked vehicles with tracking quality less than this")
    anpr_parser.add_argument("--plate_track_threshold", required=False, default="5", help="[1 ~ 9]. [1 ~ 9]. Reject tracked plates with tracking quality less than this")
    anpr_parser.add_argument("--vehicle_confidence_threshold", required=False, default="20.0", help="[0.0 ~ 100.0]. Only accept detected vehicles with confidence higher than this")
    anpr_parser.add_argument("--plate_confidence_threshold", required=False, default="15.0", help="[0.0 ~ 100.0]. Only accept detected plates with confidence higher than this")
    
    # Opencv control
    anpr_parser.add_argument("--plate_window_size", required=False, default="500", help="Plate window size for opencv in pixels. Square shaped.")
    
    # Debug output control
    anpr_parser.add_argument("--tracker_debug", required=False, default="False", help="Debug option to print trackers to terminal")
    anpr_parser.add_argument("--output_debug", required=False, default="True", help="Debug option to print csv output to terminal")
    
    anpr_args = anpr_parser.parse_args()

    return anpr_args

# Entry point
if __name__ == "__main__":

    anpr_args = get_input_args()

    #####################
    # PARAMETER CONTROL #
    #####################

    # Processing methods
    VEHICLE_DETECT_METHOD = anpr_args.vehicle_detect_method                        # Choose from: 'YOLOv5', 'ALPR_SDK'
    PLATE_DETECT_METHOD = anpr_args.plate_detect_method                            # Choose from: 'YOLOv5', 'ALPR_SDK'
    OCR_METHOD = anpr_args.ocr_detect_method                                       # Choose from: 'EasyOCR', 'PyTesseract', 'ALPR_SDK'
    PROCESSING_METHODS = [VEHICLE_DETECT_METHOD, PLATE_DETECT_METHOD, OCR_METHOD]

    # Input video paths
    video_file_name = anpr_args.video_filename                     # Choose your input video file
    video_file_extension = anpr_args.video_file_extension                       # Input video file extension type
    video_dir = anpr_args.video_file_directory     # Directory of where the video file is
    video_file_path = "".join([video_dir, video_file_name, video_file_extension])
    input_video_fps = int(anpr_args.video_fps)

    # Output mp4 video file name
    OUTPUT_VIDEO = (anpr_args.output_video == "True")                                        # [True, False]. Whether to output video.
    output_video_folder_name = "_".join(PROCESSING_METHODS)
    output_video_directory = anpr_args.output_video_directory + output_video_folder_name + '/'    # Directory to save the video file. Requires full path.
    output_video_append = anpr_args.output_video_append    # String to append to output video filename
    output_video_extension = '.mp4'
    output_video_filename = "".join([video_file_name, output_video_append, output_video_extension])
    output_video_path = "".join([output_video_directory, output_video_filename])
    output_video_fps = int(anpr_args.video_fps)                                       # +ve int. Output video frame rate. 
    output_video_resolution = (1920, 1080)                      # (width, height). Output video resolution

    # Path to save csv file
    OUTPUT_CSV = (anpr_args.output_csv == "True")                                           # [True, False]. Whether to output csv file.
    csv_output_path = "".join([output_video_directory, video_file_name, output_video_append, '.csv'])

    # ALPR-SDK parameters
    alpr_sdk_asset_path = '/home/ziv/ultimateALPR-SDK/assets'

    # Model paths
    yolov5_dir = '../yolov5'
    yolov5_weights_path = '../yolov5/vehicle_epoch_120/yolov5_vehicle_and_plate/weights/best.pt'
    haar_cascade_weights_path = 'myhaar.xml'

    # How often to perform inference
    INFERENCE_FRAME_CHECK_INTERVAL = int(anpr_args.inference_frame_check_interval)
    # How often to update tracker. Interval of frames to skip, before performing dlib tracking calculations. 
    TRACKER_FRAME_CHECK_INTERVAL = int(anpr_args.tracker_frame_check_interval)

    # Thresholds
    SPEED_LIMIT = float(anpr_args.speed_limit)                                             # +ve float or int. Speed limit of the road in km/hr to track. Minimum speed for vehicle detector to detect.
    PPM = float(anpr_args.ppm)                                                             # +ve float or int. Pixels per meter in the speed capture box.
    SLOW_MOVING_SPEED_THRESHOLD = float(5)                                                 # +ve float or int. Lower bound speed limit to count as a slow moving object. For flushing away slow moving objects that we are not interested in.
    SLOW_MOVING_TIME_THRESHOLD = float(1)                                                  # +ve float or int. How many seconds to wait before removing a slow moving object from tracking. For flushing away slow moving objects that we are not interested in.
    speedComputeBoundary = (500, 100, 1900, 800)                                    # Boundary (x_min, y_min, x_max, y_max) to perform speed calculations. Depends on input video resolution.
    DetectionBoundary = (200, 120, 1900, 920)
    VEHICLE_TRACK_THRESHOLD = int(anpr_args.vehicle_track_threshold)                                 # [1 ~ 9]. Reject tracked vehicles with tracking quality less than this.
    PLATE_TRACK_THRESHOLD = int(anpr_args.plate_track_threshold)                                   # [1 ~ 9]. Reject tracked plates with tracking quality less than this. 
    VEHICLE_CONFIDENCE_THRESHOLD = float(anpr_args.vehicle_confidence_threshold)                         # [0.0 ~ 100.0]. Only accept detected vehicles with confidence higher than this
    PLATE_CONFIDENCE_THRESHOLD = float(anpr_args.plate_confidence_threshold)                           # [0.0 ~ 100.0]. Only accept detected plates with confidence higher than this
    pd.set_option("display.max_columns", 15)
    pd.set_option('display.expand_frame_repr', False)

    ##################
    # OPENCV CONTROL #
    ##################
    titlePosition = (20, 20)                                    # Top left (x, y) of title position
    titleColour = (0, 84, 255)                                  # Title text colour
    vehicleBoxColour = (0, 255, 0)                              # Colour for vehicles
    plateBoxColour = (200, 0, 255)                              # Colour for plates
    speedComputeBoxColour = (17, 163, 252)                      # Colour for speed capture box
    plateWindowSize = int(anpr_args.plate_window_size)                                       # Plate window size for opencv in pixels. Square shaped.

    ########################
    # INITIALIZE VARIABLES #
    ########################
    frameCounter = 0
    currentCarID = 0
    currentPlateID = 0

    vehicleAttributes = {}
    plateAttributes = {}
    vehicleWithPlateAttributes = []                             # List storing the paired vehicles and plates
    vehicleWithPlate_filtered_df = pd.DataFrame()               # The pandas dataframe to eventually export as csv file

    inference_duration = None
    ocr_duration = None


    #################
    # DEBUG CONTROL #
    #################
    TRACKER_DEBUG = (anpr_args.tracker_debug == "True")
    DEBUG_OUTPUT = (anpr_args.output_debug == "True")


    ###########################
    # INITIALIZE ANPR PROGRAM #
    ###########################

    # Initialize ANPR
    init_anpr(PROCESSING_METHODS)

    # Check if file exists
    if not os.path.isfile(video_file_path):
        raise OSError("File doesn't exist: %s" % video_file_path)

    # Load video
    video = cv2.VideoCapture(video_file_path)

    if OUTPUT_VIDEO:
        # Create recursive nested directory for output video file
        Path(output_video_directory).mkdir(parents=True, exist_ok=True)

        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), output_video_fps, output_video_resolution)

    if OUTPUT_CSV:
        # Create recursive nested directory for output csv file
        Path(output_video_directory).mkdir(parents=True, exist_ok=True)

    plate_resized_image = None

    # Loop through each video frame
    while True:
        
        with time_this_code() as frame:

            # Get a frame from video
            video_ok, opencv_image = video.read()

            if type(opencv_image) == type(None):
                break

            # Decode the image to get width and height
            pil_image, imageType = load_pil_image_from_opencv_frame(opencv_image)
            width, height = pil_image.size

            # Make copy of image to draw the results on
            resultImage = opencv_image.copy()

            # Clear out trackers with poor tracking quality
            vehicleAttributes = remove_poor_trackers(opencv_image, vehicleAttributes, key='vehicle_tracker', threshold=VEHICLE_TRACK_THRESHOLD)
            plateAttributes = remove_poor_trackers(opencv_image, plateAttributes, key='plate_tracker', threshold=PLATE_TRACK_THRESHOLD)

            frameCounter = frameCounter + 1

            if not (frameCounter % INFERENCE_FRAME_CHECK_INTERVAL):

                # Inference. Get results from model.
                vehicles, plates = perform_inference(vehicle_detect_method=VEHICLE_DETECT_METHOD, plate_detect_method=PLATE_DETECT_METHOD)

                # Get the inference duration
                if function_duration:
                    inference_duration, function_duration = function_duration, None

                # If vehicles are found
                if len(vehicles):

                    # Loop through each found vehicles
                    for vehicle_idx in vehicles.index:

                        # Get vehicle_confidence
                        vehicle_confidence = vehicles.loc[vehicle_idx].get('confidence')
                        #print(f"vehicle_confidence: {vehicle_confidence}")

                        # If detected vehicle has high enough confidence
                        if (VEHICLE_DETECT_METHOD == 'haar_cascade') or (vehicle_confidence > VEHICLE_CONFIDENCE_THRESHOLD):

                            # Get vehicle_box
                            vehicle_box = vehicles.loc[vehicle_idx, ['xmin', 'ymin', 'xmax', 'ymax']]
                            vehicle_box = tuple(map(int, vehicle_box))
                            #print(f"vehicle_box: {vehicle_box}")

                            # Skip vehicle if vehicle is outside detection region
                            if not isWithinBB(vehicle_box, DetectionBoundary, threshold=0.8):
                                continue

                            # Check if new bounding box is within what is already being tracked
                            matchVehicleID = isNewBox(vehicle_box, vehicleAttributes, key='vehicle_tracker')

                            # If new vehicle is found
                            if matchVehicleID is None:

                                if TRACKER_DEBUG: print ('Creating new vehicle tracker ' + str(currentCarID))

                                # Create new tracker
                                vehicle_tracker = dlib.correlation_tracker()
                                vehicle_tracker.start_track(opencv_image, dlib.rectangle(*vehicle_box))

                                # Update attributes
                                vehicleAttributes[currentCarID] = {
                                    'vehicle_tracker': vehicle_tracker,
                                    'location1': convert_xyxy_to_xywh(*vehicle_box),
                                    'vehicle_confidence': vehicle_confidence
                                }

                                currentCarID += 1
                            
                            # If vehicle detected is already being tracked
                            else:
                                # Update attributes
                                vehicleAttributes[matchVehicleID]['vehicle_confidence'] = vehicle_confidence
                            


                # If plates are found
                if len(plates):

                    # Loop through each plate found
                    for plate_idx in plates.index:

                        # Get plate_confidence
                        plate_confidence = plates.loc[plate_idx]['confidence']
                        #print(f"plate_confidence: {plate_confidence}")

                        # If detected plate has high enough confidence
                        if plate_confidence > PLATE_CONFIDENCE_THRESHOLD:

                            # Get plate_box
                            plate_box = plates.loc[plate_idx, ['xmin', 'ymin', 'xmax', 'ymax']]
                            plate_box = tuple(map(int, plate_box))
                            #print(f"plate_box: {plate_box}")

                            # Skip plate if plate is outside detection region
                            if not isWithinBB(plate_box, DetectionBoundary, threshold=0.8):
                                continue

                            # Extract text from plate region
                            plate_number, ocr_confidence = perform_ocr(opencv_image, plate_box, ocr_method=OCR_METHOD)
                            #print(plate_number, ocr_confidence)

                            # Get the ocr duration
                            if function_duration:
                                ocr_duration, function_duration = function_duration, None

                            # Expand the size of the plate image
                            plate_resized_image = resize2SquareKeepingAspectRation(opencv_image[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]], plateWindowSize, cv2.INTER_AREA)



                            # Check if new bounding box is within what is already being tracked
                            matchPlateID = isNewBox(plate_box, plateAttributes, key='plate_tracker')

                            # If a new plate is found
                            if matchPlateID is None:

                                if TRACKER_DEBUG: print ('Creating new plate tracker ' + str(currentPlateID))

                                # Create new tracker
                                plate_tracker = dlib.correlation_tracker()
                                plate_tracker.start_track(opencv_image, dlib.rectangle(*plate_box))

                                # Update attributes
                                plateAttributes[currentPlateID] = {
                                    'plate_tracker': plate_tracker,
                                    'plate_box': plate_box,
                                    'plate_confidence': plate_confidence,
                                    'plate_number': plate_number,
                                    'ocr_confidence': ocr_confidence
                                }

                                if plate_number:
                                    
                                    # Write in the plate window
                                    cv2.putText(plate_resized_image, plate_number, (10, plateWindowSize - 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, plateBoxColour, 3)
                                    cv2.putText(plate_resized_image, "Confidence: " + str(ocr_confidence) + "%", (10, plateWindowSize - 120 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, plateBoxColour, 2)

                                currentPlateID += 1

                            # If plate detected is already being tracked
                            else:
                                # Update attributes
                                plateAttributes[matchPlateID]['plate_confidence'] = plate_confidence

                                old_plate_number = plateAttributes[matchPlateID]['plate_number']
                                old_ocr_confidence = plateAttributes[matchPlateID]['ocr_confidence']

                                # If confidence for plate number has increased
                                if plate_number and ((old_ocr_confidence is None) or (ocr_confidence > old_ocr_confidence)):

                                    # Update attributes
                                    plateAttributes[matchPlateID]['plate_number'] = plate_number
                                    plateAttributes[matchPlateID]['ocr_confidence'] = ocr_confidence

                                    # Write in the plate window
                                    cv2.putText(plate_resized_image, plate_number, (10, plateWindowSize - 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, plateBoxColour, 3)
                                    cv2.putText(plate_resized_image, "Confidence: " + str(ocr_confidence) + "%", (10, plateWindowSize - 120 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, plateBoxColour, 2)

                                # If confidence for plate number has not increased
                                elif old_plate_number:

                                    # Write in the plate window
                                    cv2.putText(plate_resized_image, old_plate_number, (10, plateWindowSize - 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, plateBoxColour, 3)
                                    cv2.putText(plate_resized_image, "Confidence: " + str(old_ocr_confidence) + "%", (10, plateWindowSize - 120 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, plateBoxColour, 2)

            # Track all vehicles
            resultImage, vehicleAttributes = trackAllVehicles(resultImage, vehicleAttributes, colour=vehicleBoxColour)

            # Track all plates
            resultImage, plateAttributes = trackAllPlates(resultImage, plateAttributes, colour=plateBoxColour)

            # Compute speed for each tracked vehicle
            resultImage, vehicleAttributes = get_vehicle_speed(resultImage, vehicleAttributes, frameCounter, boundary=speedComputeBoundary)

            # Overlay resized plate window with resulting image
            if plate_resized_image is not None:
                resultImage[0:plateWindowSize, 0:plateWindowSize] = plate_resized_image

            # Draw dotted bounding box for where speed is measuring
            drawrect(resultImage, (speedComputeBoundary[0], speedComputeBoundary[1]), (speedComputeBoundary[2], speedComputeBoundary[3]), speedComputeBoxColour, thickness=3, style='dotted')
            cv2.putText(resultImage, "Speed Capture Area", (speedComputeBoundary[0], speedComputeBoundary[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, speedComputeBoxColour, 2)

            # Write inference methods on video
            cv2.putText(resultImage, "Vehicle Detector: " + VEHICLE_DETECT_METHOD, (titlePosition[0], titlePosition[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, speedComputeBoxColour, 2)
            cv2.putText(resultImage, "  Plate Detector: " + PLATE_DETECT_METHOD, (titlePosition[0], titlePosition[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, speedComputeBoxColour, 2)
            cv2.putText(resultImage, "   OCR Detector: " + OCR_METHOD, (titlePosition[0], titlePosition[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, speedComputeBoxColour, 2)



            ##################
            # DATA FILTERING #
            ##################

            # Match the plate to the vehicle
            frame_plate_vehicle_pair = match_plate_with_vehicle(vehicleAttributes, plateAttributes, threshold=0.1)

            # Add the new unique plate and vehicle pairs to the list
            new_plate_vehicle_pair = [x for x in frame_plate_vehicle_pair if x not in vehicleWithPlateAttributes]
            vehicleWithPlateAttributes.extend(new_plate_vehicle_pair)

            # Convert to pandas dataframe
            frame_plate_vehicle_pair_df = pd.DataFrame.from_records(frame_plate_vehicle_pair)
            vehicleWithPlate_df = pd.DataFrame.from_records(vehicleWithPlateAttributes)

            # Filter out unnecessary columns
            drop_column_list = ['vehicle_tracker', 'plate_tracker', 'location2', 'speed_captured', 'slow_vehicle_counter']
            frame_plate_vehicle_pair_filtered_df = frame_plate_vehicle_pair_df.drop(columns=drop_column_list, errors='ignore')
            vehicleWithPlate_filtered_df = vehicleWithPlate_df.drop(columns=drop_column_list, errors='ignore')

            # Filter out unnecessary (nearly duplicate) rows
            drop_duplicates_column_list = ['vehicle_id', 'plate_id', 'plate_number']
            # Keep the row with highest ocr_confidence for matching vehicle and plates
            if 'ocr_confidence' in vehicleWithPlate_filtered_df.columns:
                vehicleWithPlate_filtered_df = vehicleWithPlate_filtered_df.sort_values(by='ocr_confidence', axis=0, ascending=True, ignore_index=True)
                vehicleWithPlate_filtered_df.drop_duplicates(subset=drop_duplicates_column_list, keep='last', inplace=True)
                
                # Re-order the columns
                column_order = ['vehicle_id', 'plate_id', 'speed', 'plate_number', 'ocr_confidence', 'vehicle_confidence', 'plate_confidence', 'location1', 'plate_box']
                vehicleWithPlate_filtered_df = vehicleWithPlate_filtered_df[column_order]

            if DEBUG_OUTPUT and (not (frameCounter % 120)) and len(vehicleWithPlate_filtered_df):
                print("\n\n\n")
                print("Vehicles and plates data after " + str(frameCounter) + " frames: ")
                print(vehicleWithPlate_filtered_df)

                # Write to csv file. No compression to save faster.
                if OUTPUT_CSV and len(vehicleWithPlate_filtered_df): 
                    vehicleWithPlate_filtered_df.to_csv(csv_output_path, sep=',', index=False, header=True, encoding='utf-8', compression=None)

        # Write inference_duration, ocr_duration, and process_duration
        if inference_duration:
            cv2.putText(resultImage, " Inference time: " + str(inference_duration) + " ms", (titlePosition[0], titlePosition[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, speedComputeBoxColour, 2)
        if ocr_duration:
            cv2.putText(resultImage, "      OCR time: " + str(ocr_duration) + " ms", (titlePosition[0], titlePosition[1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, speedComputeBoxColour, 2)
        if frame.time:
            cv2.putText(resultImage, "    Frame time: " + str(frame.time) + " ms", (titlePosition[0], titlePosition[1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, speedComputeBoxColour, 2)

        # Write the frame into output video file
        if OUTPUT_VIDEO:
            out.write(resultImage)



        # Show image
        cv2.imshow('ANPR',resultImage)

        if cv2.waitKey(33) == 27:
            break

    # Sort rows by highest speed to lowest speed
    if 'speed' in vehicleWithPlate_filtered_df.columns:
        vehicleWithPlate_filtered_df = vehicleWithPlate_filtered_df.sort_values(by='speed', axis=0, ascending=False, ignore_index=True)

        # Print final output to terminal
        print("\n\n\n")
        print("Vehicles and plates data after " + str(frameCounter) + " frames: ")
        print(vehicleWithPlate_filtered_df)
    
        # Write to csv file
        if OUTPUT_CSV and len(vehicleWithPlate_filtered_df): 
            vehicleWithPlate_filtered_df.to_csv(csv_output_path, sep=',', index=False, header=True, encoding='utf-8')
    


    cv2.destroyAllWindows()

    # DeInit the ALPR-SDK engine
    if 'ALPR_SDK' in PROCESSING_METHODS: 
        checkResult("DeInit", ultimateAlprSdk.UltAlprSdkEngine_deInit())
