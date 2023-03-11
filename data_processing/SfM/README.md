## SfM

The code is used for getting actions from RGB videos (SfM). 

Download meshroom binary file from [link](https://www.fosshub.com/Meshroom.html?dwl=Meshroom-2021.1.0-linux-cuda10.tar.gz)
### Split Video into frames:
<pre>
python split_video.py
</pre>
### Mask Gripper (optional):
<pre>
python mask.py
</pre>
### SfM:
<pre>
python data_processing.py
</pre>
You can find the camera pose estimations from "/cache/StructureFromMotion/cameras.sfm"
### Remove failures:
<pre>
python check.py
</pre>
### Visualization:
<pre>
python visualize_labels.py
</pre>