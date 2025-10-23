#!/bin/bash
runs=( 1234 1231 1235 1232 1236 1237 1238 1239 1240 1241 1243 1245 1246 1248 1249 1250 1252 1267 1268 1270 1271 1272 1273 1274 1275 1261 1260 )
for run in "${runs[@]}"
do
    echo "run${run}"
    python3 filterFERSForCalibration.py --runNumbers ${run} --muonveto --holeveto --PSD --CC1
done
