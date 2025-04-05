export PYTHONPATH=/home/uky/repos_python/Research/PIBO:$PYTHONPATH

# # TODO: Add the ablation for the Main Components A.2.1 Priority 1
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#         --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --buffer_size 300 --alpha 1e-4 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#         --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --ablation Default&
# done
# wait
# for seed in 0 1 2 3; do # No Filtering
#     CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#         --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --buffer_size 300 --alpha 1e-4 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#         --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering False --ablation MainComponents&
# done
# wait
# for seed in 0 1 2 3; do # No Filtering, Local Search
#     CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#         --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#         --buffer_size 300 --alpha 1e-4 --local_search False --local_search_epochs 0 --diffusion_steps 30\
#         --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering False --ablation MainComponents&
# done
# wait
# for seed in 0 1 2 3; do #No Filtering, Local Search, Posterior training
#     CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#         --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 0\
#         --buffer_size 300 --alpha 1e-4 --local_search False --local_search_epochs 0 --diffusion_steps 30\
#         --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering False --ablation MainComponents&
# done
# wait

# TODO: Add the ablation for the Reweighting Scheme A.1.1 Priority 1
# for reweighting in uniform value rank; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-4 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting $reweighting --filtering True --ablation Reweighting&
#     done
#     wait
# done
# wait

# #TODO: Add the ablation for the alpha A.2.2 Priority 1
# for alpha in 1e-5 1e-3 1e-2 1e-1 1e0; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha $alpha --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --ablation Alpha&
#     done
#     wait
# done
# wait

# #TODO: Add the ablation for the buffer size A.3.1 Priority 1
# for buffer_size in 100 200 300 500 1000 2000; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size $buffer_size --alpha 1e-4 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --ablation Buffer&
#     done
#     wait
# done
# wait

# #TODO: Add the ablation for the uncertainty estimation A.1.3 Priority 2
# for uncertainty_estimation in dropout; do #None; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-4 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 0.1 --num_ensembles 5 --reweighting exp --filtering True --uncertainty_estimation $uncertainty_estimation\
#             --ablation UncertaintyEstimation&
#     done
#     wait
# done
# wait

# for uncertainty_estimation in dropout None; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task RoverPlanning --dim 100 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-5 --local_search True --local_search_epochs 30 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --uncertainty_estimation $uncertainty_estimation\
#             --ablation UncertaintyEstimation&
#     done
#     wait
# done
# wait

# for uncertainty_estimation in dropout None; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task DNA --dim 180 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-5 --local_search True --local_search_epochs 50 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --uncertainty_estimation $uncertainty_estimation\
#             --ablation UncertaintyEstimation&
#     done
#     wait
# done
# wait

# #TODO: Add the ablation for the gamma A.1.4 Priority 2
# for gamma in 0 0.1 0.5 3.0 5.0 10.0; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-4 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma $gamma --num_ensembles 5 --reweighting exp --filtering True --ablation Gamma&
#     done
#     wait
# done
# wait
# #TODO: Add the ablation for the gamma A.1.4 Priority 2
# for gamma in 0 0.1 0.5 3.0 5.0 10.0; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task RoverPlanning --dim 100 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-5 --local_search True --local_search_epochs 30 --diffusion_steps 30\
#             --gamma $gamma --num_ensembles 5 --reweighting exp --filtering True --ablation Gamma&
#     done
#     wait
# done
# wait
# # #TODO: Add the ablation for the gamma A.1.4 Priority 2
# for gamma in 0 0.1 0.5 3.0 5.0 10.0; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task DNA --dim 180 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-5 --local_search True --local_search_epochs 50 --diffusion_steps 30\
#             --gamma $gamma --num_ensembles 5 --reweighting exp --filtering True --ablation Gamma&
#     done
#     wait
# done
# wait

# #TODO: Add the ablation for the local search epochs A.2.3 Priority 2
# for local_search_epochs in 0 30 50 100; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-4 --local_search True --local_search_epochs $local_search_epochs --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --ablation LocalSearchEpochs&
#     done
#     wait
# done
# wait

#TODO: Add the ablation for the Amortized Sampler A.2.5 Priority 2
# for training_posterior in on; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-4 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --training_posterior $training_posterior\
#             --ablation AmortizedSampler&
#     done
#     wait
# done
# wait

#TODO: Add the ablation for the Experiment Setting A.3.2 Priority 2
# for n_init in 200 500 1000; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#             --n_init $n_init --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-4 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --ablation NInit&
#     done
#     wait
# done
# wait

# for batch_size in 10 100 200 400; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size $batch_size\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-4 --local_search True --local_search_epochs 10 --diffusion_steps 30\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --ablation BatchSize&
#     done
#     wait
# done
# wait

# #TODO: Add the ablation for the training steps A.1.2 Priority 3
for numepoch in 30; do #100 150; do
    for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
            --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs $numepoch --num_prior_epochs $numepoch\
            --num_posterior_epochs $numepoch --buffer_size 300 --alpha 1e-4 --local_search True --local_search_epochs 10\
            --diffusion_steps 30 --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --ablation TrainingSteps&
    done
    wait
done
wait

# #TODO: Add the ablation for the diffusion steps A.3.3 Priority 3
# for diffusion_steps in 10 50 100; do
#     for seed in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/pibo.py --task HalfCheetah --dim 102 --batch_size 50\
#             --n_init 100 --seed $seed --max_evals 2000 --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 50\
#             --buffer_size 300 --alpha 1e-4 --local_search True --local_search_epochs 10 --diffusion_steps $diffusion_steps\
#             --gamma 1.0 --num_ensembles 5 --reweighting exp --filtering True --ablation DiffusionSteps&
#     done
#     wait
# done
# wait
