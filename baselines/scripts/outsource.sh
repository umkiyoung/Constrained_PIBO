export PYTHONPATH=/home/uky/repos_python/Research/Constrained_PIBO:$PYTHONPATH

for seed in 0; do
   CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/outsource.py --task Ackley --dim 200 --batch_size 100\
       --n_init 200 --max_evals 10000 --seed $seed --num_proxy_epochs 50 --num_prior_epochs 50 --num_posterior_epochs 100\
       --lamb 1.0 --constraint_formulation None --flow_steps 30 --buffer_size 500\
       --langevin --local_search --pis_architectures --mode_fwd tb --mode_bwd tb --zero_init --clipping&
done