export PYTHONPATH=/home/uky/repos_python/Research/Constrained_PIBO:$PYTHONPATH

# Train script
CUDA_VISIBLE_DEVICES=3 python baselines/algorithms/diffusion_sampler_viz.py \
    --t_scale 5. \
    --energy 25gmm \
    --pis_architectures \
    --zero_init \
    --clipping \
    --mode_fwd tb \
    --lr_policy 1e-3 \
    --lr_back 1e-3 \
    --lr_flow 1e-1 \
    --exploratory \
    --exploration_wd \
    --exploration_factor 0.1 \
    --both_ways \
    --local_search \
    --buffer_size 600000 \
    --prioritized rank \
    --rank_weight 0.01 \
    --ld_step 0.1 \
    --ld_schedule \
    --target_acceptance_rate 0.574