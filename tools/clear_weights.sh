#!/bin/bash 

# 需要删除的文件夹路径 
folder_path="data"

# 递归删除文件夹下所有以'.pt'结尾的文件
function delete_pt_files(){
    for f in "$1"/*
    do
        if [ -d "$f" ]   # 如果是目录,递归搜索
        then  
            delete_pt_files "$f" 
        elif [[ $f == *.pt ]] # 如果是pt文件,删除
        then
            rm "$f"
        fi
    done
}

delete_pt_files "$folder_path"

echo "deleted '.pt' files" 
echo "This is for recalculate isg and ist weight"
