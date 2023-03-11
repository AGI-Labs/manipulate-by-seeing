#!/usr/bin/env python3


import pyrealsense2 as rs


realsense_ctx = rs.context()
connected_devices = []
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    connected_devices.append(detected_camera)
print(connected_devices)