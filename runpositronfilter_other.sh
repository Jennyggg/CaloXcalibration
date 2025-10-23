#!/bin/bash
runs=( 1261 1260 )
for run in "${runs[@]}"
do
    echo "run${run}"
    python3 filterFERSForCalibration.py --runNumbers ${run} --muonveto --PSD --CC1
done
