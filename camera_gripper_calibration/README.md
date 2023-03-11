## Calibration

The code is used for hand eye calibration. The estimated actions are in the tool frame, while the robot command is in the robot frame. We need to run hand eye calibration to align these two frames. 

### Collect data for calibration (Record Camera Pose and Robot Pose):
<pre>
python calibration_record.py
</pre>
Please move the robot for 1 minute.
### Run Calibration:
<pre>
python compute_calibration.py
</pre>
Notations: sr -> spatial frame of robot; br -> body frame of robot; st -> spatial frame of tool; bt -> body frame of tool. After running calibration, you will get T_br_bt.

T_br0br1 = T_br_bt * T_bt0bt1 * (T_br_bt)^-1