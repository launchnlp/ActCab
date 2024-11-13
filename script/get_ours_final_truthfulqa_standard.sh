#!/bin/bash

set -e

#
declare -A models

#
models[Llama-2-7b-hf,NQ]=31
models[Llama-2-7b-hf,sciq]=31
models[Llama-2-7b-hf,truthfulqa_r]=31
models[Llama-2-7b-hf,truthfulqa_standard]=31
models[Llama-2-7b-hf,triviaqa]=31
models[Llama-2-7b-hf,wikiqa]=31

models[Llama-2-13b-hf,NQ]=39
models[Llama-2-13b-hf,sciq]=39
models[Llama-2-13b-hf,truthfulqa_r]=39
models[Llama-2-13b-hf,truthfulqa_standard]=39
models[Llama-2-13b-hf,triviaqa]=39
models[Llama-2-13b-hf,wikiqa]=39

models[Meta-Llama-3-8B,NQ]=31
models[Meta-Llama-3-8B,sciq]=31
models[Meta-Llama-3-8B,truthfulqa_r]=31
models[Meta-Llama-3-8B,truthfulqa_standard]=31
models[Meta-Llama-3-8B,triviaqa]=31
models[Meta-Llama-3-8B,wikiqa]=31

models[Llama-2-7b-chat-hf,NQ]=31
models[Llama-2-7b-chat-hf,sciq]=31
models[Llama-2-7b-chat-hf,truthfulqa_r]=31
models[Llama-2-7b-chat-hf,truthfulqa_standard]=31
models[Llama-2-7b-chat-hf,triviaqa]=31
models[Llama-2-7b-chat-hf,wikiqa]=31
# meta-llama/Llama-2-7b-chat-hf
#
get_value() {
    local model=$1
    local dataset=$2
    local key="${model},${dataset}"
    local value=${models[$key]}

    if [ -z "$value" ]; then
        echo "Unknown dataset: $dataset for model: $model"
        return 1
    fi

    echo $value
}

datasets=("truthfulqa_standard")
# datasets=("wikiqa")
model_name=$1
num_samples=1
model_name_postfix=$(echo "$model_name" | cut -d'/' -f2)
for dataset in "${datasets[@]}"; do
    echo "dataset: $dataset"
    cd ../src

    h_layer=$(get_value $model_name_postfix $dataset)

    echo h_layer $h_layer

    logits_file="../${dataset}/${model_name_postfix}_ICL_w_answer_hidden_states_4_answer_newsamples_normal_demos_sequence_num_demos5.pt"
    if [ ! -f $logits_file ]; then
        echo "collecting logits for sampled generations"
        python collect_logits_for_ICL_w_answer_confidence_4_answer_normal_demos_sequence_truthfulqa_standard.py ${model_name} ${dataset} 4 5
    fi

    # 3. train linear layers
    echo "training linear layers"
    calibrator_seq_avg_ckpt="../${dataset}/ICL_cls_ckpt/${model_name_postfix}_calibrator_intenvention_seq_avg.pt"
    python train_ICL_seq_avg.py ${model_name} ${dataset} sample 5

    # 1. sample model answers
    model_gen_file="../${dataset}/test${model_name_postfix}.num_demos_5.greedy_search.csv"
    echo "generating model answers"
    # if [ ! -f $model_gen_file ]; then
    python gen_model_answers_final_truthfulqa_standard.py $model_name $dataset $num_samples test 5
    # fi

    # 2. estimate the correctness of vanilla greedy search answers
    echo "estimating the correctness of vanilla greedy search answers"
    correctness_file="../${dataset}/test${model_name_postfix}.num_demos_5.greedy_search.IK_set.csv"
    # if [ ! -f $correctness_file ]; then
    python get_triviaqa_IK_training_set_final.py ${dataset} test $model_gen_file
    # fi


    # 3. estimate the confidence of vanilla greedy search answers
    echo "estimating the confidence of vanilla greedy search answers"
    confidence_file="../${dataset}/test${model_name_postfix}.num_demos_5.greedy_search.IK_set.confidence_seq_avg_greedy_linear.csv"
    # if [ ! -f $confidence_file ]; then
    python gen_conf_collect_logits_for_ICL_w_answer_confidence_4_answer_normal_demos_avg_final_truthfulqa_standard.py $model_name $dataset $h_layer $correctness_file $model_gen_file 5
    # fi

    # 4. guided greedy search
    guided_model_gen_file="../${dataset}/test${model_name_postfix}.num_demos_5.greedy_search.fact_guided_seq_avg_greedy_linear.csv"
    echo "guided greedy search"
    # if [ ! -f $guided_model_gen_file ]; then
    python gen_model_answers_fact_guided_seq_avg_final_truthfulqa_standard.py $model_name $dataset $h_layer $num_samples 5
    # fi

    # 5. estimate the correctness of guided greedy search answers
    echo "estimating the correctness of guided greedy search answers"
    guided_correctness_file="../${dataset}/test${model_name_postfix}.num_demos_5.greedy_search.fact_guided_seq_avg_greedy_linear.IK_set.csv"
    # if [ ! -f $guided_correctness_file ]; then
    python get_triviaqa_IK_training_set_final.py ${dataset} test $guided_model_gen_file
    # fi

    # 6. estimate the confidence of guided greedy search answers
    echo "estimating the confidence of guided greedy search answers"
    guided_confidence_file="../${dataset}/test${model_name_postfix}.num_demos_5.greedy_search.fact_guided_seq_avg_greedy_linear.IK_set.confidence_seq_avg_greedy_linear.csv"
    # if [ ! -f $guided_confidence_file ]; then
    python gen_conf_collect_logits_for_ICL_w_answer_confidence_4_answer_normal_demos_avg_final_truthfulqa_standard.py $model_name $dataset $h_layer $guided_correctness_file $guided_model_gen_file 5
    # fi

    # 7. get the final results
    echo "get the final results"
    python filter_generation.py $confidence_file $guided_confidence_file "../${dataset}/test.txt"
done