#! /bin/bash
for i in {0..19}
do
    python3 sample_image.py --seed $i
done