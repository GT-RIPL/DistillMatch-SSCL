# sh experiments/TinyIMNET_20k_uniform_20.sh 
# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=20

###############################################################
# save directory
OUTDIR=outputs/TinyIMNET-20k/uniform_${SPLIT}
MAXTASK=-1

# hard coded inputs
REPEAT=1
SCHEDULE="200"
MODELNAME="WideResNet_28_2_tinyimnet"
MODELNAMEOOD_DC="WideResNet_DC_28_2_tinyimnet"
L_DIST="vanilla"
BS=32
UBS=64
WD=5e-4

# GD parameters
SCHEDULE_GD="120 160 180 200"

# uniform specific parameters
MEMORY=0

# tuned paramaters
LR_PL=0.1
WA_PL=1
TPR=0.5
TPR_OOD=0.5
LR_GD=0.005
Co_GD=1.0

###############################################################

# process inputs
mkdir -p $OUTDIR

# dm
python -u run_sscl.py --dataset TinyIMNET --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 20000 --unlabeled_task_samples -1 \
    --force_out_dim 200 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
    --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
    --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
    --weight_aux $WA_PL --fm_loss  \
    --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --DW --FT  \
    --tpr $TPR --oodtpr $TPR_OOD  \
    --max_task $MAXTASK --log_dir ${OUTDIR}/dm

# oracle
python -u run_sscl.py --dataset TinyIMNET --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 20000 --unlabeled_task_samples -1 \
    --force_out_dim 200 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
    --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD \
    --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC   --model_type resnet \
    --learner_type distillmatch --learner_name DistillMatch --tpr $TPR --oodtpr $TPR_OOD --oracle_flag \
    --max_task $MAXTASK --log_dir ${OUTDIR}/oracle

# base
python -u run_sscl.py --dataset TinyIMNET --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 20000 --unlabeled_task_samples -1 \
    --force_out_dim 200 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
    --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD \
    --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC  --model_type resnet \
    --learner_type distillmatch --learner_name DistillMatch --tpr $TPR --oodtpr $TPR_OOD \
    --max_task $MAXTASK --log_dir ${OUTDIR}/base

# distillation and retrospection
python -u run_sscl.py --dataset TinyIMNET --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 20000 --unlabeled_task_samples -1 \
        --force_out_dim 200 --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
        --optimizer SGD --lr $LR_GD  --momentum 0.9 --weight_decay $WD   \
        --memory $MEMORY --model_name $MODELNAME --model_type resnet  \
        --learner_type distillation --learner_name GD --co 0.0 --distill_loss L C    \
        --max_task $MAXTASK --log_dir ${OUTDIR}/dr

# global distillation
python -u run_sscl.py --dataset TinyIMNET --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 20000 --unlabeled_task_samples -1 \
    --force_out_dim 200 --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
    --optimizer SGD --lr $LR_GD  --momentum 0.9 --weight_decay $WD   \
    --memory $MEMORY --model_name $MODELNAME --model_type resnet --DW --FT \
    --learner_type distillation --learner_name GD --co $Co_GD --distill_loss P C Q    \
    --max_task $MAXTASK --log_dir ${OUTDIR}/gd

# end to end incremental learning
python -u run_sscl.py --dataset TinyIMNET --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 20000 --unlabeled_task_samples -1 \
        --force_out_dim 200 --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
        --optimizer SGD --lr $LR_GD  --momentum 0.9 --weight_decay $WD   \
        --memory $MEMORY --model_name $MODELNAME --model_type resnet --DW --FT \
        --learner_type distillation --learner_name GD --co 0.0 --distill_loss L   \
        --max_task $MAXTASK --log_dir ${OUTDIR}/ete


    
