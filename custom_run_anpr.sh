cd ./anpr

LD_LIBRARY_PATH=../alpr_sdk:$LD_LIBRARY_PATH \
python3 anpr.py \
--vehicle_detect_method 'YOLOv5' \
--plate_detect_method 'YOLOv5' \
--ocr_detect_method 'PyTesseract' \
--video_filename 'traffic_departing_3' \
--video_file_extension '.mov' \
--video_file_directory '/home/ziv/speeding_catcher/traffic_footage/' \
--video_fps 30 \
--output_video True \
--output_video_directory '/home/ziv/speeding_catcher/output_video/' \
--output_video_append '_output' \
--output_csv True \
--inference_frame_check_interval 1 \
--tracker_frame_check_interval 1 \
--speed_limit 2 \
--vehicle_track_threshold 8 \
--plate_track_threshold 6 \
--vehicle_confidence_threshold 20.0 \
--plate_confidence_threshold 15.0 \
--plate_window_size 600 \
--tracker_debug False \
--output_debug True

