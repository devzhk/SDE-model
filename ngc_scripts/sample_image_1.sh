#! /bin/bash
for i in {40..44}
do
    python3 sample_image.py --seed $i
done