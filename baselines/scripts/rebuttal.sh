export PYTHONPATH=/home/uky/repos_python/Research/PIBO:$PYTHONPATH

# Grid search over n_init and batch_size
n_init_list="10 50"
batch_size_list="20 50 200"
default_n_init=200
default_batch_size=100

# Define tasks
tasks="Rastrigin"

# #NOTE: ------------------------------------------------------------TurBO-------------------------------------------------------------

# # Grid search over n_init with default batch_size
# for task in $tasks; do
#     for n_init in $n_init_list; do
#         for seed in 0 1 2 3; do
#             CUDA_VISIBLE_DEVICES=$seed \
#             python baselines/algorithms/mcmcbo.py --task $task --dim 200 --tr_num 1 \
#                 --batch_size $default_batch_size --noise_var 0 --n_init $n_init --max_evals 10000 --seed $seed \
#                 --use_mcmc 0 --repeat_num 1 &
#         done
#         wait
#     done
# done

# # Grid search over batch_size with default n_init
# for task in $tasks; do
#     for batch_size in $batch_size_list; do
#         for seed in 0 1 2 3; do
#             CUDA_VISIBLE_DEVICES=$seed \
#             python baselines/algorithms/mcmcbo.py --task $task --dim 200 --tr_num 1 \
#                 --batch_size $batch_size --noise_var 0 --n_init $default_n_init --max_evals 10000 --seed $seed \
#                 --use_mcmc 0 --repeat_num 1 &
#         done
#         wait
#     done
# done

# #NOTE: ------------------------------------------------------------DiffBBO-------------------------------------------------------------
# Grid search over n_init with default batch_size
# for task in $tasks; do
#     for n_init in $n_init_list; do
#         for seed in 0 1 2 3; do
#             CUDA_VISIBLE_DEVICES=$seed \
#             python baselines/algorithms/diffbbo.py --task $task --dim 200 --batch_size 100 \
#                 --n_init $n_init --max_evals 11000 --seed $seed &
#         done
#         wait
#     done
# done

# # Grid search over batch_size with default n_init
# for task in $tasks; do
#     for batch_size in $batch_size_list; do
#         for seed in 0 1 2 3; do
#             CUDA_VISIBLE_DEVICES=$seed \
#             python baselines/algorithms/diffbbo.py --task $task --dim 200 --batch_size $batch_size \
#                 --n_init 200 --max_evals 10000 --seed $seed &
#         done
#         wait
#     done
# done

# #NOTE: ------------------------------------------------------------PIBO n_init=50-------------------------------------------------------------
# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task Rastrigin --dim 200 --batch_size 100\
#        --n_init 50 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#        --local_search True --alpha 1e-5 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 1000 &
# done

# wait

# #NOTE: ------------------------------------------------------------PIBO n_init=10-------------------------------------------------------------

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task Rastrigin --dim 200 --batch_size 100\
#        --n_init 10 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#        --local_search True --alpha 1e-5 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 1000 &
# done

# wait

for n_ensemble in 1 2 3 4 5 6 7 8 9 10; do
    for seed in 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$((seed-4)) python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
        --n_init 100 --max_evals 2000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
        --local_search True --alpha 1e-4 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 300 --num_ensemble $n_ensemble &
    done
    wait
done
wait
