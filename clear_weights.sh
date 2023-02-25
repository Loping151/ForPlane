#!/bin/bash

# 需要删除的文件夹路径
folder_path="data"

# 删除文件夹下所有以'.pt'结尾的文件
find "$folder_path" -type f -name '*.pt' -delete

echo "deleted '.pt' files"
echo "This is for recalculate isg and ist weight"

