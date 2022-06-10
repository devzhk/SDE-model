#! /bin/bash
for i in {20..39}
do
    python3 sample_image.py --seed $i
done