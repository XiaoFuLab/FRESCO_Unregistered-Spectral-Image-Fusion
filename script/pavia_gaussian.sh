dataroot=data/pavia_clean_gaussian.mat
resultroot=result/clean_version/clean_pavia_gaussian


python code/Main_two_stage.py\
    --dataRoot $dataroot\
    --resultRoot $resultroot/\
    --firstStageResult $resultroot/decomposition_result.mat\
    --numMaterial 4\
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
    --RGB '52,30,7'\
    --overlap 11\
    --wandb False\
    --project_name MELD\
    --textLog True\
    --run_name pavia_gaussian\

