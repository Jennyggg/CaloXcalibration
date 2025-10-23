#!/bin/bash
runs=( 1422 1423 1424 )
for run in "${runs[@]}"
do
    echo "run${run}"
    python3 filterFERSForCalibration.py --runNumbers ${run} --muoncounter --PSD --CC1 --CC2 --CC3
    python3 filterFERSForCalibration.py --runNumbers ${run} --holeveto --muoncounter --PSD --CC1 --CC2 --CC3
done
