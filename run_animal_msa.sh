CUDA_VISIBLE_DEVICES=0 python happier/run.py \
'experience.experiment_name=MSA_dyml_animal' \
'experience.log_dir=experiments' \
experience.seed=0 \
experience.warmup_step=40 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0] \
experience.accuracy_calculator.overall_accuracy=True \
experience.accuracy_calculator.exclude=[NDCG,H-AP] \
experience.accuracy_calculator.recall_rate=[10,20] \
experience.accuracy_calculator.with_binary_asi=True \
optimizer=animal_sgd \
model=animal_r34_multi_emb \
transform=dyml \
dataset=dyml_animal \
loss=ClusterLoss_MultiEmb
