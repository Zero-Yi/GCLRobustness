#!/bin/bash

# # 设置运行次数
# num_runs=5

# # 循环运行指定次数
# for ((i=1; i<=num_runs; i++))
# do
#     python MyGCL.py   # 运行你的 Python 程序
# done
python dgi.py --dataset Cora --PGD True --PRBCD True > log/DGI/Cora.log 
python dgi.py --dataset CiteSeer --PGD True --PRBCD True > log/DGI/CiteSeer.log 
python dgi.py --dataset PubMed --PGD True --PRBCD True > log/DGI/PubMed.log 