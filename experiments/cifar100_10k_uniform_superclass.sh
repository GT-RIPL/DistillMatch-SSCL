# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
SPLIT=5

###############################################################
# save directory
OUTDIR=outputs/CIFAR100-10k/uniform_superclass
MAXTASK=-1

# hard coded inputs
REPEAT=1
SCHEDULE="200"
MODELNAME="WideResNet_28_2_cifar"
MODELNAMEOOD_DC="WideResNet_DC_28_2_cifar"
L_DIST="super"
UL_DIST="vanilla"
BS=64
UBS=128
WD=5e-4

# GD parameters
SCHEDULE_GD="120 160 180 200"

# realistic specific parameters
MEMORY=400

# tuned paramaters
LR_PL=0.1
WA_PL=1
TPR=0.05
TPR_OOD=0.05
LR_GD=0.1
Co_GD=1.0

###############################################################

# process inputs
mkdir -p $OUTDIR

# dm
python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --ul_dist $UL_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 \
    --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS    \
    --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD   \
    --learner_type distillmatch --learner_name DistillMatch --pl_flag  \
    --weight_aux $WA_PL --fm_loss \
    --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC --model_type resnet --DW --FT  \
    --tpr $TPR --oodtpr $TPR_OOD \
    --max_task $MAXTASK --log_dir ${OUTDIR}/dm

# base
python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 \
    --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT  --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
    --optimizer SGD --lr $LR_PL --momentum 0.9 --weight_decay $WD \
    --memory $MEMORY --model_name $MODELNAME --ood_model_name $MODELNAMEOOD_DC  --model_type resnet \
    --learner_type distillmatch --learner_name DistillMatch --tpr $TPR --oodtpr $TPR_OOD \
    --max_task $MAXTASK --log_dir ${OUTDIR}/base
    
# end to end incremental learning
python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --ul_dist $UL_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 \
        --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
        --optimizer SGD --lr $LR_GD  --momentum 0.9 --weight_decay $WD   \
        --memory $MEMORY --model_name $MODELNAME --model_type resnet --DW --FT \
        --learner_type distillation --learner_name GD --co 0.0 --distill_loss L   \
        --max_task $MAXTASK --log_dir ${OUTDIR}/ete

# global distillation
python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --ul_dist $UL_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 \
    --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
    --optimizer SGD --lr $LR_GD  --momentum 0.9 --weight_decay $WD   \
    --memory $MEMORY --model_name $MODELNAME --model_type resnet --DW --FT \
    --learner_type distillation --learner_name GD --co $Co_GD --distill_loss P C Q    \
    --max_task $MAXTASK --log_dir ${OUTDIR}/gd

# distillation and retrospection
python -u run_sscl.py --dataset CIFAR100 --l_dist $L_DIST --ul_dist $UL_DIST --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --labeled_samples 10000 --unlabeled_task_samples -1 \
        --force_out_dim 100 --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE_GD --schedule_type decay --batch_size $BS --ul_batch_size $UBS   \
        --optimizer SGD --lr $LR_GD  --momentum 0.9 --weight_decay $WD   \
        --memory $MEMORY --model_name $MODELNAME --model_type resnet  \
        --learner_type distillation --learner_name GD --co 0.0 --distill_loss L C    \
        --max_task $MAXTASK --log_dir ${OUTDIR}/dr