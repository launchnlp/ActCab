set -e

datasets=("NQ" "sciq" "truthfulqa_r" "triviaqa" "wikiqa")
model_name=$1
# mistralai/Mistral-7B-v0.1
# meta-llama/Meta-Llama-3-8B
# google/gemma-7b
num_samples=4
model_name_postfix=$(echo "$model_name" | cut -d'/' -f2)
for dataset in "${datasets[@]}"; do
    echo "dataset: $dataset"

    cd ../src

    # 1. sample model answers
    model_gen_file="../${dataset}/train${model_name_postfix}temperature_1topp_1.0_num_demos_20_answers.${num_samples}.csv"
    echo "sampling model answers for training linear layers"
    if [ ! -f $model_gen_file ]; then
        python gen_model_answers.py $model_name $dataset $num_samples train
    fi

    # 2. get correctness of sampled generations
    echo "getting correctness of sampled generations"
    model_gen_file="../${dataset}/train${model_name_postfix}temperature_1topp_1.0_num_demos_20_answers.${num_samples}.csv"
    correctness_file="../${dataset}/train${model_name_postfix}temperature_1topp_1.0_num_demos_20_answers.${num_samples}.IK_set.csv"
    if [ ! -f $correctness_file ]; then
        python get_triviaqa_IK_training_set.py ${dataset} train $model_gen_file
    fi
    
    # 3. Collect logits for sampled generations
    logits_file="../${dataset}/${model_name_postfix}_ICL_w_answer_hidden_states_4_answer_newsamples_normal_demos_sequence_true.pt"
    if [ ! -f $logits_file ]; then
        echo "collecting logits for sampled generations"
        python collect_logits_for_ICL_w_answer_confidence_4_answer_normal_demos_sequence.py ${model_name} ${dataset} ${num_samples}
    fi

    # 3. train linear layers
    echo "training linear layers"
    # python train_ICL.py ${model_name} ${dataset}
    calibrator_seq_ckpt="../${dataset}/ICL_cls_ckpt/${model_name_postfix}_calibrator_intenvention_seq.pt"
    if [ ! -f $calibrator_seq_ckpt ]; then
        python train_ICL_seq.py ${model_name} ${dataset}
    fi
    calibrator_seq_avg_ckpt="../${dataset}/ICL_cls_ckpt/${model_name_postfix}_calibrator_intenvention_seq_avg.pt"
    if [ ! -f $calibrator_seq_avg_ckpt ]; then
        python train_ICL_seq_avg.py ${model_name} ${dataset}
    fi

    # # 1. gen model answers
    # echo "generating model answers"
    # python gen_model_answers.py $model_name $dataset 1

    # 2. get confidence for 
done