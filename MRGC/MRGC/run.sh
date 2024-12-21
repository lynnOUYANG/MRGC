#!/bin/bash



#n_T=(1 2 4 6 8)
#dim_yelp=(4 8 16 32 64)
#seed=( 2 4 6 8 9 10 15 20 25 30 35 40 42 43 44 45 46 48 50 52 55 58 60 64)
#alpha=(1)
#sigma=(0.05 0.06 0.08)
#gamma=(0. 0.1 0.2 0.4 0.01 0.02 0.04 0.08 0.001 0.005 0.008 0.0001 0.0005 0.0008)
#theta=(200 400 600 800 1000 1200 1500 2000 3000 3500 4000 5000 6000 8000 10000)
#for alpha in "${alpha[@]}"; do
#  for sigma in "${sigma[@]}"; do
#    for n_T in "${n_T[@]}"; do
#      python main.py  --n_T $n_T --alpha $alpha --fusion False --dataset amazon --gamma 0. --dim 65 --sigma $sigma --theta 1 --seed 42 --gpu 1
#    done
#  done
#done


python main.py  --n_T 6 --alpha 1 --fusion False --dataset acm-3025 --gamma 0. --dim 128 --sigma 3 --theta 160 --seed 6 --max_h 2
python main.py  --n_T 4 --alpha 1 --fusion False --dataset dblp --gamma 0. --dim 64 --sigma 80 --theta 2 --seed 42 --max_h 1
python main.py  --n_T 6 --alpha 1 --fusion False --dataset acm-4019 --gamma 0.1 --dim 512 --sigma 5 --theta 200 --seed 6 --max_h 3
python main.py  --n_T 10 --alpha 1 --fusion False --dataset yelp --gamma 0. --dim 33 --sigma 34 --theta 60 --seed 6 --max_h 3
python main.py  --n_T 4 --alpha 1 --fusion False --dataset amazon --gamma 0. --dim 65 --sigma 1 --theta 15 --seed 42 --gpu 1 --max_h 3
python main.py  --n_T 18 --alpha 1 --fusion False --dataset imdb --gamma 0.005 --dim 1024 --sigma 9 --theta 80 --seed 42 --max_h 1
#
python large.py --n_T 10 --alpha 1 --sigma 4 --dataset mag --gamma 0. --dim 64 --theta 100 --seed 42 --gpu 1 --max_h 40

#