#!/bin/bash




VIDEO_FILE_NAMES=('traffic_both_1' 'traffic_both_2' 'traffic_departing_1' 'traffic_departing_2' 'traffic_departing_3' 'traffic_departing_4' 'traffic_oncoming_1' 'traffic_oncoming_2')

#VIDEO_FILE_NAMES=('traffic_both_1' 'traffic_both_2' 'traffic_departing_1' 'traffic_departing_2')




# Batch run in subshells
for VIDEO_FILE_NAME in "${VIDEO_FILE_NAMES[@]}"; do

    gnome-terminal --title=${VIDEO_FILE_NAME} --geometry 0x0+0+850 --hide-menubar -- bash -c "./subshell_batch_run_anpr.sh ${VIDEO_FILE_NAME}"
    
done

echo "END OF $(basename "$0") SCRIPT"

#LD_LIBRARY_PATH=../alpr_sdk:$LD_LIBRARY_PATH \
#python3 anpr.py \
#--vehicle_detect_method $VEHICLE_DETECT_METHOD \
#--plate_detect_method $PLATE_DETECT_METHOD \
#--ocr_detect_method $OCR_DETECT_METHOD \
#--video_filename $VIDEO_FILE_NAME \
#--inference_frame_check_interval 1 \
#--tracker_frame_check_interval 5 \
#--speed_limit 2 \
#--ppm 82.5 \
#--vehicle_track_threshold 8 \
#--plate_track_threshold 5 \
#--vehicle_confidence_threshold 20.0 \
#--plate_confidence_threshold 15.0



