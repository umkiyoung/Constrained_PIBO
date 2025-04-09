export PYTHONPATH=/home/uky/repos_python/Research/Constrained_PIBO:$PYTHONPATH

for seed in 0; do
    CUDA_VISIBLE_DEVICES=2 python baselines/algorithms/scbo.py --task Ackley --dim 10 --batch_size 1\
        --n_init 10 --max_evals 500 --seed $seed 
done
wait