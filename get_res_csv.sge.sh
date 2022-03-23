#!/bin/bash

#$ -o $JOB_NAME.o$JOB_ID
#$ -N pack_results
EXPERIMENTS="focal200 focal200_ovassion"
# EXPERIMENTS=$EXPERIMENTS" focal200reinpatho focal200reinpatho_ovassion "
EXPERIMENTS=$EXPERIMENTS" focal200_1_ovassion focal200reinpatho_1_ovassion "

# EXPERIMENTS=$EXPERIMENTS" focal200_2_ovassion focal200_3_ovassion focal200_5_ovassion focal200_6_ovassion" 
# EXPERIMENTS=$EXPERIMENTS" focal200_7_ovassion focal200_8_ovassion focal200_9_ovassion focal200_10_ovassion"
# EXPERIMENTS=$EXPERIMENTS" focal200reinpatho_2_ovassion focal200reinpatho_3_ovassion focal200reinpatho_5_ovassion focal200reinpatho_6_ovassion" 
# EXPERIMENTS=$EXPERIMENTS" focal200reinpatho_7_ovassion focal200reinpatho_8_ovassion focal200reinpatho_9_ovassion focal200reinpatho_10_ovassion"

#$ -q volta.q
#$ -l gpu=1

##export PATH="/usr/local/bin:/sbin:$PATH"
export PATH="/usr/local/bin:$PATH"


singularity exec --nv ../tensorflow-gpu.simg python res_to_csv.py $EXPERIMENTS