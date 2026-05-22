dataroot=data/terrain_nearest.mat
resultroot=result/terrain_nearest


python code/Main_two_stage.py\
    --dataRoot $dataroot\
    --resultRoot $resultroot/\
    --firstStageResult $resultroot/decomposition_result.mat\
    --numMaterial 5\
    --Stage1 True\
    --Stage2 True\
    --cuda True\
    --sr 4\
    --iterationCBTD 400\
    --s2o_weight 1e-4\
    --nEpochs 2000\
    --batchSize 8\
    --iterEachEpoch 20\
    --patchSize 12\
    --lr 1e-4\
    --decayEpoch 1000\
    --sideSampling 0.2\
    --scale_weight 15\
    --gan_weight 1\
    --inverse_weight 10\
    --scaleUp 2.0\
    --eval_Stage1 True\
    --eval_Stage2 True\
    --eval_interval 100\
    --FID True\
    --RGB '38,30,15'\
    --overlap 11\
    --wandb True\
    --project_name MELD\
    --textLog True\
    --run_name terrain_nearest\

    
