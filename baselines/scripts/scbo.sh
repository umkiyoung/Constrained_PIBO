export PYTHONPATH=/home/uky/repos_python/Research/Constrained_PIBO:$PYTHONPATH

for seed in 0; do
    CUDA_VISIBLE_DEVICES=2 python baselines/algorithms/scbo.py --task Ackley --dim 200 --batch_size 100\
        --n_init 200 --max_evals 10000 --seed $seed 
done
wait