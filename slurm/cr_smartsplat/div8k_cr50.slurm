#!/bin/bash
#SBATCH -J SmartSplatExp_DIV8K_CR50
#SBATCH -p A800
#SBATCH -N 1                  # 1个节点
#SBATCH --gres=gpu:a800:1     # 每个任务1块GPU（关键修改）
#SBATCH -n 1                  # 每个任务分配4个CPU核（按需调整）
#SBATCH -o slurm_logs/SmartSplatExp_DIV8K_CR50.out # %A为作业ID，%a为数组任务ID
#SBATCH -e slurm_logs/SmartSplatExp_DIV8K_CR50.err

# 加载环境
source /share/home/uxxx/miniconda3/etc/profile.d/conda.sh
module purge
module load cuda/12.4
module load gcc/11.4.0
conda activate smartsplat

# 检查GPU
nvidia-smi || echo "GPU检测失败"

# pip install scikit-learn
# pip install matplotlib

# 打印PyTorch环境信息
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# 数据路径和日志
data_path="/share/home/uxxx/workspace/datahome/DIV8K"
base_log_path="/share/home/uxxx/workspace/datahome/smartsplat_logs"

current_cr=50
# 启动训练（单GPU模式）
srun python train_all_for_eval_smart.py \
    -d $data_path \
    --data_name  DIV8K \
    --model_name SmartSplat \
    --compression_ratio $current_cr \
    --log_dir $base_log_path \
    --iterations 50000 \
    --save_iter_img 10000