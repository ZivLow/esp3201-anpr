cd ./anpr

LD_LIBRARY_PATH=../alpr_sdk:$LD_LIBRARY_PATH \
python3 anpr.py \
--vehicle_detect_method 'YOLOv5' \
--plate_detect_method 'YOLOv5' \
--ocr_detect_method 'EasyOCR' \
--video_filename 'traffic_departing_1' \
--inference_frame_check_interval 1 \
--tracker_frame_check_interval 1 \
--speed_limit 2 \
--vehicle_track_threshold 8 \
--plate_track_threshold 6 \
--vehicle_confidence_threshold 20.0 \
--plate_confidence_threshold 15.0

