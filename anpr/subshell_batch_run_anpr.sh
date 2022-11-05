#!/bin/bash

# FILENAME			PPM value
# 'traffic_both_1'		28.5
# 'traffic_both_2'		28.5
# 'traffic_departing_1'		44.6
# 'traffic_departing_2'		22.45
# 'traffic_departing_3'		33.0
# 'traffic_departing_4'		22.5
# 'traffic_oncoming_1'		33.5
# 'traffic_oncoming_2'		33.5

# Array of PPM values for different video files
declare -A ppm_value=(
    [traffic_both_1]="28.5"
    [traffic_both_2]="28.5"
    [traffic_departing_1]="40.0"
    [traffic_departing_2]="22.45"
    [traffic_departing_3]="33.0"
    [traffic_departing_4]="22.5"
    [traffic_oncoming_1]="33.5"
    [traffic_oncoming_2]="33.5"
    )

#VEHICLE_DETECT_METHODS=('haar_cascade' 'YOLOv5' 'ALPR_SDK')
VEHICLE_DETECT_METHODS=('haar_cascade')
#PLATE_DETECT_METHODS=('YOLOv5' 'ALPR_SDK')
PLATE_DETECT_METHODS=('ALPR_SDK')
OCR_DETECT_METHODS=('EasyOCR' 'PyTesseract' 'ALPR_SDK')
	    
VIDEO_FILE_NAME=$1

for VEHICLE_DETECT_METHOD in "${VEHICLE_DETECT_METHODS[@]}"; do
    for PLATE_DETECT_METHOD in "${PLATE_DETECT_METHODS[@]}"; do
        for OCR_DETECT_METHOD in "${OCR_DETECT_METHODS[@]}"; do
        

            LD_LIBRARY_PATH=../alpr_sdk:$LD_LIBRARY_PATH \
            python3 anpr.py \
            --vehicle_detect_method $VEHICLE_DETECT_METHOD \
            --plate_detect_method $PLATE_DETECT_METHOD \
            --ocr_detect_method $OCR_DETECT_METHOD \
            --video_filename $VIDEO_FILE_NAME \
            --inference_frame_check_interval 30 \
            --tracker_frame_check_interval 1 \
            --speed_limit 5 \
            --ppm ${ppm_value[${VIDEO_FILE_NAME}]} \
            --vehicle_track_threshold 8 \
            --plate_track_threshold 5 \
            --vehicle_confidence_threshold 20.0 \
            --plate_confidence_threshold 15.0
            
            echo
            echo "Completed $VEHICLE_DETECT_METHOD $PLATE_DETECT_METHOD $OCR_DETECT_METHOD detection methods for $VIDEO_FILE_NAME"
            
            echo
            sleep 3
            
        done
    done
done

