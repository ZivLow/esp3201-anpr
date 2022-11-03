#!/bin/bash

cd ./anpr

VEHICLE_DETECT_METHODS=('YOLOv5' 'ALPR_SDK')
PLATE_DETECT_METHODS=('YOLOv5' 'ALPR_SDK')
OCR_DETECT_METHODS=('EasyOCR' 'PyTesseract' 'ALPR_SDK')
VIDEO_FILE_NAMES=('traffic_both_1' 'traffic_both_2' 'traffic_departing_1' 'traffic_departing_2' 'traffic_departing_3' 'traffic_oncoming_1' 'traffic_oncoming_2')

for VEHICLE_DETECT_METHOD in "${VEHICLE_DETECT_METHODS[@]}"; do
    for PLATE_DETECT_METHOD in "${PLATE_DETECT_METHODS[@]}"; do
        for OCR_DETECT_METHOD in "${OCR_DETECT_METHODS[@]}"; do
            for VIDEO_FILE_NAME in "${VIDEO_FILE_NAMES[@]}"; do
                LD_LIBRARY_PATH=../alpr_sdk:$LD_LIBRARY_PATH \
python3 anpr.py \
--vehicle_detect_method $VEHICLE_DETECT_METHOD \
--plate_detect_method $PLATE_DETECT_METHOD \
--ocr_detect_method $OCR_DETECT_METHOD \
--video_filename $VIDEO_FILE_NAME \
--inference_frame_check_interval 1 \
--tracker_frame_check_interval 1 \
--speed_limit 2 \
--vehicle_track_threshold 8 \
--plate_track_threshold 6 \
--vehicle_confidence_threshold 20.0 \
--plate_confidence_threshold 15.0

                echo "Completed $VEHICLE_DETECT_METHOD $PLATE_DETECT_METHOD $OCR_DETECT_METHOD detection methods for $VIDEO_FILE_NAME"
                
                done
            done
        done
    done
done





