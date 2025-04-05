export PYTHONPATH=/home/uky/repos_python/Research/PIBO:$PYTHONPATH

#!/usr/bin/env bash

###############################################################################
# PIBO
###############################################################################
for task in HalfCheetah; do
    for seed in 8 9; do
        CUDA_VISIBLE_DEVICES=$((seed-8)) python baselines/algorithms/pibo.py --task "$task" --dim 102 \
            --batch_size 50 --n_init 100 --max_evals 2000 --seed "$seed" \
            --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50 \
            --local_search True --alpha 1e-4 --local_search_epochs 10 --gamma 1.0 --buffer_size 300 &
    done
done


for task in RoverPlanning; do
    for seed in 8 9; do
        CUDA_VISIBLE_DEVICES=$((seed-6)) python baselines/algorithms/pibo.py --task "$task" --dim 100 \
            --batch_size 50 --n_init 100 --max_evals 2000 --seed "$seed" \
            --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50 \
            --local_search True --alpha 1e-5 --local_search_epochs 30 --gamma 1.0 --buffer_size 300 &
    done
done
wait

for task in DNA; do
    for seed in 8 9; do
        CUDA_VISIBLE_DEVICES=$((seed-8)) python baselines/algorithms/pibo.py --task "$task" --dim 180 \
            --batch_size 50 --n_init 100 --max_evals 2000 --seed "$seed" \
            --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50 \
            --local_search True --alpha 1e-5 --local_search_epochs 50 --gamma 1.0 --buffer_size 300 &
    done
done

###############################################################################
# DDOM
###############################################################################
for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-6)) python baselines/algorithms/ddom.py --task HalfCheetah --dim 102 --batch_size 50 \
        --n_init 100 --max_evals 2000 --seed "$seed" &
done
wait

for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-8)) python baselines/algorithms/ddom.py --task RoverPlanning --dim 100 --batch_size 50 \
        --n_init 100 --max_evals 2000 --seed "$seed" &
done


for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-6)) python baselines/algorithms/ddom.py --task DNA --dim 180 --batch_size 50 \
        --n_init 100 --max_evals 2000 --seed "$seed" &
done
wait

###############################################################################
# DiffBBO
###############################################################################
for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-8)) python baselines/algorithms/diffbbo.py --task HalfCheetah --dim 102 --batch_size 50 \
        --n_init 100 --max_evals 2000 --seed "$seed" &
done


for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-6)) python baselines/algorithms/diffbbo.py --task RoverPlanning --dim 100 --batch_size 50 \
        --n_init 100 --max_evals 2000 --seed "$seed" &
done
wait

for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-8)) python baselines/algorithms/diffbbo.py --task DNA --dim 180 --batch_size 50 \
        --n_init 100 --max_evals 2000 --seed "$seed" &
done


###############################################################################
# MINs
# --num_epochs $num_epochs 부분은 환경 변수/혹은 스크립트 내 선언을 가정합니다.
###############################################################################
for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-6)) python baselines/algorithms/mins.py --task HalfCheetah --dim 102 \
        --n_init 100 --batch_size 50 --max_evals 2000 --seed "$seed" --num_epochs "$num_epochs" &
done
wait

for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-8)) python baselines/algorithms/mins.py --task RoverPlanning --dim 100 \
        --n_init 100 --batch_size 50 --max_evals 2000 --seed "$seed" --num_epochs "$num_epochs" &
done


for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-6)) python baselines/algorithms/mins.py --task DNA --dim 180 \
        --n_init 100 --batch_size 50 --max_evals 2000 --seed "$seed" --num_epochs "$num_epochs" &
done
wait

###############################################################################
# CMA-ES
###############################################################################
for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-8)) python baselines/algorithms/cmaes.py --task HalfCheetah --dim 102 --batch_size 50 \
        --n_init 100 --max_evals 2000 --seed "$seed" &
done


for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-6)) python baselines/algorithms/cmaes.py --task RoverPlanning --dim 100 --batch_size 50 \
        --n_init 100 --max_evals 2000 --seed "$seed" &
done
wait

for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-8)) python baselines/algorithms/cmaes.py --task DNA --dim 180 --batch_size 50 \
        --n_init 100 --max_evals 2000 --seed "$seed" &
done


###############################################################################
# MCMC-BO
###############################################################################
for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-6)) python baselines/algorithms/mcmcbo.py --task HalfCheetah --dim 102 --tr_num 1 --batch_size 50 \
        --noise_var 0 --n_init 100 --max_evals 2000 --seed "$seed" --use_mcmc MH --repeat_num 1 &
done
wait

for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-8)) python baselines/algorithms/mcmcbo.py --task RoverPlanning --dim 100 --tr_num 1 --batch_size 50 \
        --noise_var 0 --n_init 100 --max_evals 2000 --seed "$seed" --use_mcmc MH --repeat_num 1 &
done


for seed in 8 9; do
    CUDA_VISIBLE_DEVICES=$((seed-6)) python baselines/algorithms/mcmcbo.py --task DNA --dim 180 --tr_num 1 --batch_size 50 \
        --noise_var 0 --n_init 100 --max_evals 2000 --seed "$seed" --use_mcmc MH --repeat_num 1 &
done
wait
