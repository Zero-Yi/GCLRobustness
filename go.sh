#!/bin/bash

# # 设置运行次数
# num_runs=5

# # 循环运行指定次数
# for ((i=1; i<=num_runs; i++))
# do
#     python MyGCL.py   # 运行你的 Python 程序
# done
python gcn.py --dataset Cora --PRBCD True --GRBCD True --normalize True > log/GCN/Cora.log 
python gcn.py --dataset CiteSeer --PRBCD True --GRBCD True --normalize True > log/GCN/CiteSeer.log
python gcn.py --dataset PubMed --PRBCD True --GRBCD True --normalize True > log/GCN/PubMed.log