export PYTHONPATH=/home/uky/repos_python/Research/PIBO:$PYTHONPATH

# for seed in 0 1 2 3; do
#    CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/tb.py --task Ackley --dim 200 --batch_size 100\
#        --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 250\
#        --local_search True --alpha 1e-5 --local_search_epochs 50 --diffusion_steps 30 --buffer_size 2000&
# done



for seed in 0 1 2 3; do
   CUDA_VISIBLE_DEVICES=$((seed)) python baselines/algorithms/tb.py --task Rastrigin --dim 200 --batch_size 100\
       --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 250\
       --local_search True --alpha 1e-5 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 1000 &
done

wait

for seed in 4 5 6 7; do
   CUDA_VISIBLE_DEVICES=$((seed-4)) python baselines/algorithms/tb.py --task Rastrigin --dim 200 --batch_size 100\
       --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 250\
       --local_search True --alpha 1e-5 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 1000 &
done

wait

for seed in 0 1 2 3; do
   CUDA_VISIBLE_DEVICES=$((seed)) python baselines/algorithms/tb.py --task HalfCheetah --dim 102 --batch_size 50\
       --n_init 100 --max_evals 2000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 250\
       --local_search True --alpha 1e-4 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 300 &
done

wait

for seed in 4 5 6 7; do
   CUDA_VISIBLE_DEVICES=$((seed-4)) python baselines/algorithms/tb.py --task HalfCheetah --dim 102 --batch_size 50\
       --n_init 100 --max_evals 2000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 250\
       --local_search True --alpha 1e-4 --local_search_epochs 10 --diffusion_steps 30 --buffer_size 300 &
done