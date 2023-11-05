if [ 1 -eq 1 ];then
CUDA_VISIBLE_DEVICES=0 python happier/run.py \
'experience.experiment_name=MSA_dyml_product' \
'experience.log_dir=experiments' \
experience.seed=0 \
experience.max_iter=20 \
experience.warmup_step=0 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0,1,2] \
experience.accuracy_calculator.overall_accuracy=True \
experience.accuracy_calculator.exclude=[NDCG,H-AP] \
experience.accuracy_calculator.recall_rate=[10,20] \
experience.accuracy_calculator.with_binary_asi=True \
optimizer=product_sgd \
model=product_r34_single_emb \
transform=dyml \
dataset=dyml_product \
loss=ClusterLoss_SingleEmb
fi

if [ 1 -eq 0 ];then
CUDA_VISIBLE_DEVICES=0 python happier/run.py \
'experience.experiment_name=MSA_dyml_product' \
'experience.log_dir=experiments' \
experience.seed=0 \
experience.max_iter=20 \
experience.warmup_step=0 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0,1,2] \
experience.accuracy_calculator.overall_accuracy=True \
experience.accuracy_calculator.exclude=[NDCG,H-AP] \
experience.accuracy_calculator.recall_rate=[10,20] \
experience.accuracy_calculator.with_binary_asi=True \
optimizer=product_sgd1 \
model=product_r34_multi_emb \
transform=dyml \
dataset=dyml_product \
loss=ClusterLoss_MultiEmb
fi
