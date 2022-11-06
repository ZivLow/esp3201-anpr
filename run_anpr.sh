cd ./anpr

LD_LIBRARY_PATH=../alpr_sdk:$LD_LIBRARY_PATH \
python3 anpr.py \
--vehicle_detect_method 'haar_cascade' \
--plate_detect_method 'ALPR_SDK' \
--ocr_detect_method 'EasyOCR' \
--video_filename 'traffic_departing_2' \
--video_file_directory '/home/ziv/speeding_catcher/traffic_footage/' \
--output_video False \
--output_video_directory '/home/ziv/speeding_catcher/output_video_demo/' \
--output_csv False \
--inference_frame_check_interval 30 \
--tracker_frame_check_interval 1 \
--speed_limit 2 \
--ppm 25.0 \
--vehicle_track_threshold 8 \
--plate_track_threshold 6 \
--vehicle_confidence_threshold 20.0 \
--plate_confidence_threshold 15.0 \
--plate_window_size 500 

