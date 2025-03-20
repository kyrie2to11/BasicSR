#!/bin/bash

echo "此脚本作用是删除 .png 文件的中间路径,将 .png 文件移动到上一级目录 e.g. LSDIR_train_HR/0001000/0000001.png -> LSDIR_train_HR/0000001.png"
echo "请输入 LSDIR 数据集所在路径(最好用绝对路径):"
read target_path

# 切换到目标路径
cd "${target_path}" || exit 1
echo "current working directory = $(pwd)"

# 遍历目标路径下的所有子目录 
for dir in "${target_path}/"*/; do 
    # 判断是否为目录 
    echo "${dir}"
    if [ -d "${dir}" ]; then 
        # 移动子目录下的 .png 文件到目标路径 
        mv "${dir}"*.png "${target_path}"
    fi
done 

# 删除空的子目录 
find "${target_path}" -mindepth 1 -type d -empty -delete

echo ".png 文件移动成功且空子目录已删除"