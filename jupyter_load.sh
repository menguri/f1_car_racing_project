#!/bin/bash
#SBATCH --nodes=3
#SBATCH --partition=gpu1 
#SBATCH --cpus-per-task=56 
#SBATCH --gres=gpu:4
#SBATCH --job-name=UBAIJOB    
#SBATCH -o ./jupyter_output/jupyter.%N.%j.out  # STDOUT ( Standard Output ) 
#SBATCH -e ./jupyter_output/jupyter.%N.%j.err  # STDERR ( Standard Error ) 

echo "start at:" `date` # 접속한 날짜 표기
echo "node: $HOSTNAME" # 접속한 노드 번호 표기 
echo "jobid: $alsrn5051" # jobid 표기 

# GPU 환경을 이용하고 싶은 경우에만 해당! 그렇지 않은 경우 해당 명령어들은 지우셔도 무관합니다.
# module avail CUDA # CUDA 어떤 버전들이 설치되어있는지 확인하는 방법 
# module unload CUDA/11.2.2 # 기본적으로 탑재되어있는 쿠다버전은 unload. 
# module load cuda/11.8.0 # GPU를 사용하는 경우 CUDA 버전을 지정해 줄 수 있으며, UBAI 최신 CUDA 버전은 12.2.1입니다.
module python/3.11.2 # GPU를 사용하는 경우 CUDA 버전을 지정해 줄 수 있으며, UBAI 최신 CUDA 버전은 12.2.1입니다.


python -m jupyter lab $HOME \
        --ip=0.0.0.0 \
        --no-browse