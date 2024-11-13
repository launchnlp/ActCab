set -e

datasets=("NQ" "sciq" "truthfulqa_r" "triviaqa" "wikiqa")
model_name=$1
num_samples=4
model_name_postfix=$(echo "$model_name" | cut -d'/' -f2)
for dataset in "${datasets[@]}"; do
    echo "dataset: $dataset"

    cd ../src

    # 1. sample model answers
    model_gen_file="../${dataset}/train${model_name_postfix}_num_demos_5_answers.${num_samples}.paired.csv"
    echo "sampling model answers for training linear layers"
    python gen_model_paired_samples.py $model_name $dataset $num_samples train 5


    # 2. get correctness of sampled generations
    echo "getting correctness of sampled generations"
    model_gen_file="../${dataset}/train${model_name_postfix}_num_demos_5_answers.${num_samples}.paired.csv"
    python get_triviaqa_IK_training_set.py ${dataset} train $model_gen_file

done