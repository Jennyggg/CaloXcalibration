#!/bin/bash

runs=( 1442 1441 1439 1438 1437 1429 1431 1433 1434)
energies=( 10 20 30 40 60 80 100 120 160 )

array_length=${#runs[@]}
for (( i=0; i<array_length; i++ ));
do
    echo "run ${runs[i]}"
    echo "energy ${energies[i]}"
    python3 filterFERSForCalibration_forML.py --runNumbers ${runs[i]} --energies ${energies[i]} --muoncounter --PSD --CC1 --CC2 --CC3 --particle pion --holeveto
done
