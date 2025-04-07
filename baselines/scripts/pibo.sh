export PYTHONPATH=/home/uky/repos_python/Research/Constrained_PIBO:$PYTHONPATH

#-----------------------------


for seed in 0 1 2 3; do
   CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task Ackley --dim 10 --batch_size 1\
       --n_init 10 --max_evals 200 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
       --local_search True --alpha 1e-4 --lamb 1.0 --constraint_formulation Lagrangian --local_search_epochs 10 --diffusion_steps 30 --buffer_size 500&
done



wait
# #Synthetic

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task Ackley --dim 200 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#        --local_search True --alpha 1e-5 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 500 &
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task Ackley --dim 400 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 100 --num_prior_epochs 100 --num_posterior_epochs 100\
#        --local_search True --alpha 1e-5 --local_search_epochs 15 --diffusion_steps 30 --buffer_size 500 --proxy_hidden_dim 512 &
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task Rastrigin --dim 200 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#        --local_search True --alpha 1e-5 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 1000 &
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task Rastrigin --dim 400 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 100 --num_prior_epochs 100 --num_posterior_epochs 100\
#        --local_search True --alpha 1e-5 --local_search_epochs 15 --diffusion_steps 30 --buffer_size 1000 --proxy_hidden_dim 512 &
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task Rosenbrock --dim 200 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#        --local_search True --alpha 1e-5 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 500 &
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task Rosenbrock --dim 400 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 100 --num_prior_epochs 100 --num_posterior_epochs 100\
#        --local_search True --alpha 1e-5 --local_search_epochs 15 --diffusion_steps 30 --buffer_size 500 --proxy_hidden_dim 512 &
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task Levy --dim 200 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#        --local_search True --alpha 1e-5 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 500 &
# done

# wait

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task Levy --dim 400 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 100 --num_prior_epochs 100 --num_posterior_epochs 100\
#        --local_search True --alpha 1e-5 --local_search_epochs 15 --diffusion_steps 30 --buffer_size 500 --proxy_hidden_dim 512 &
# done

# wait

# HalfCheetah
# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#        --n_init 100 --max_evals 2000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#        --local_search True --alpha 1e-4 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 300 &
# done

# wait

# RoverPlanning
# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task RoverPlanning --dim 100 --batch_size 50\
#        --n_init 100 --max_evals 2000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#        --local_search True --alpha 1e-5 --local_search_epochs 30 --diffusion_steps 30 --buffer_size 300 &
# done

# wait

# DNA
# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task DNA --dim 180 --batch_size 50\
#        --n_init 100 --max_evals 2000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#        --local_search True --alpha 1e-5 --local_search_epochs 50 --diffusion_steps 30 --buffer_size 300 &
# done

# wait