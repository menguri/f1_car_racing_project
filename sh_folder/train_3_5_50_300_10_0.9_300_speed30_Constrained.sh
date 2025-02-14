#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu1 
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --job-name=MINGK    
#SBATCH -o ./jupyter_output/jupyter.%N.%j.out  # STDOUT ( Standard Output ) 
#SBATCH -e ./jupyter_output/jupyter.%N.%j.err  # STDERR ( Standard Error ) 

echo "start at:" `date`
echo "node: $HOSTNAME"
echo "jobid: $alsrn5051"

module python/3.11.2

# 파라미터 값 정의
TU=3
TC=5
TR=50
MAP_RESET_INTERVAL=300
REPLAY_RATIO=10
EPISODES=1000
ALPHA=0.9
SAVE_POINT=100
ENV_COUNT=4
MAX_TIMESTEP=900
CONSTRAINED_USE=1

## Constrained RL
CONSTRAINED_USE=1
MAX_SPEED=30

# train.py 실행 시 파라미터 전달
python ../model_train.py --TU $TU \
                  --TC $TC \
                  --TR $TR \
                  --map_reset_interval $MAP_RESET_INTERVAL \
                  --replay_ratio $REPLAY_RATIO \
                  --episodes $EPISODES \
                  --alpha $ALPHA \
                  --save_point $SAVE_POINT \
                  --env_count $ENV_COUNT \
                  --max_timestep $MAX_TIMESTEP \
                  --constrained_use $CONSTRAINED_USE \
                  --max_speed $MAX_SPEED