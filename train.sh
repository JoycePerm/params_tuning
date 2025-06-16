#!/bin/bash

DATA_PATH_0="/home/chenjiayi/code/params-tuning/data/type0/data_normalized.pt"
DATA_PATH_1="/home/chenjiayi/code/params-tuning/data/type1/data_normalized.pt"
OUTPUT_PATH_0="/home/chenjiayi/code/params-tuning/log/type0_LN"
OUTPUT_PATH_1="/home/chenjiayi/code/params-tuning/log/type1_LN"
PYTHON_SCRIPT="/home/chenjiayi/code/params-tuning/train.py"

echo "请选择要执行的数据集:"
echo "0) data0"
echo "1) data1"
read -p "输入数字(0/1): " choice

# 检查输入是否合法
if [[ "$choice" != "0" && "$choice" != "1" ]]; then
    echo "错误:请输入0或1"
    exit 1
fi

if [ "$choice" -eq 0 ]; then
    echo "训练数据集0"
    DATA_PATH=$DATA_PATH_0
    OUTPUT_PATH=$OUTPUT_PATH_0
else
    echo "训练数据集1"
    DATA_PATH=$DATA_PATH_1
    OUTPUT_PATH=$OUTPUT_PATH_1
fi

# DATA_PATH="/home/chenjiayi/code/params-tuning/data/all_data_normalized.pt"
# OUTPUT_PATH="/home/chenjiayi/code/params-tuning/log/type"
# PYTHON_SCRIPT="/home/chenjiayi/code/params-tuning/train.py"

mkdir -p "$OUTPUT_PATH"

echo "TRAINING ..."

# echo "$DATA_PATH"   
python "$PYTHON_SCRIPT" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_PATH" \
    > "$OUTPUT_PATH/train_loss.log" 2>&1

echo "Loss results saved to $OUTPUT_PATH/train_loss.log"
echo "TRAINING FINISHED!..."