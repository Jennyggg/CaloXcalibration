#!/bin/bash
runs=( 1442 1441 1439 1438 1437 1429 1431 )
energies=( 10 20 30 40 60 80 100 )
array_length=${#runs[@]}
for (( i=0; i<array_length; i++ ));
do
    echo "run ${runs[i]}"
    echo "energy ${energies[i]}"
    python3 filterFERSForCalibration.py --runNumbers ${runs[i]} --energies ${energies[i]}  --holeveto --muoncounter --PSD --CC1 --CC2 --CC3 --isHadron
done
