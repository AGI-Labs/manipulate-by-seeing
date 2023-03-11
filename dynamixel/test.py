from dynamixel_py import dxl
import numpy as np
import time


ACTUATOR_IDS=[1]

gripper = dxl(motor_id=ACTUATOR_IDS, motor_type='X', baudrate=57600, devicename='/dev/ttyUSB0')
gripper.open_port()
gripper.engage_motor(motor_id=ACTUATOR_IDS, enable=True)

gripper.set_des_pos(ACTUATOR_IDS, [np.pi])
time.sleep(5)
gripper.set_des_pos(ACTUATOR_IDS, [np.deg2rad(300)])

time.sleep(5)
gripper.engage_motor(motor_id=ACTUATOR_IDS, enable=False)
