#!/bin/bash
# 设置要检查的GPU ID
gpu_ids=(0 1 2 3 5 6 7)

# 等待时间变量
wait_time=5  # 每次等待的时间（秒）
total_waited=0  # 累计等待时间（秒）

# 检查GPU的占用情况
while true; do
    available_gpus=()

    for gpu_id in "${gpu_ids[@]}"; do
        # 获取当前GPU的已用内存和总内存
        memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
        memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu_id)
        # 计算内存占用的百分比
        memory_percentage=$((memory_used * 100 / memory_total))

        # 判断GPU内存占用是否小于40%
        if [ "$memory_percentage" -lt 10 ]; then
            available_gpus+=($gpu_id)
        fi
    done

    # 如果GPU数组长度等于2，运行Python脚本
    if [ ${#available_gpus[@]} -ge 1 ]; then
        echo "找到可用的GPU: ${available_gpus[@]}，正在运行程序..."

        for i in {1..100000}
        do
            echo "Running iteration $i"
            CUDA_VISIBLE_DEVICES=${available_gpus[0]} python /mnt/d/models/StoryDiffusion/Comic_Generation.py --seed=114514 --para="amuse"
        done

        break
    else
        # 显示累计等待时间
        echo "所有指定的GPU都被占用，已等待 $total_waited 秒，等待 $wait_time 秒后重试..."
        sleep 1  # 等待指定的时间
        total_waited=$((total_waited + wait_time))  # 累加等待时间
        # sleep 2  # 等待10秒后重试
    fi
done
