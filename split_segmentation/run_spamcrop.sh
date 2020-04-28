rm data/cropped_image.npy
rm data/merged_image.npy
python3 splitandmerge.py
python3 splitandcrop.py
python2 spamcrop_visualize.py
