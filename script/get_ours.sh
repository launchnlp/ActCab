set -e

# datasets=("NQ" "sciq" "truthfulqa_r" "triviaqa" "wikiqa")
datasets=("NQ" "sciq")
model_name=$1
num_samples=1
model_name_postfix=$(echo "$model_name" | cut -d'/' -f2)
for dataset in "${datasets[@]}"; do
    echo "dataset: $dataset"
    cd ../src

    # 1. sample model answers
    model_gen_file="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_20_answers.greedy_search.csv"
    echo "generating model answers"
    if [ ! -f $model_gen_file ]; then
        python gen_model_answers.py $model_name $dataset $num_samples test
    fi
    # python gen_model_answers.py $model_name $dataset $num_samples test

    # 2. estimate the correctness of vanilla greedy search answers
    echo "estimating the correctness of vanilla greedy search answers"
    correctness_file="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_20_answers.greedy_search.IK_set.csv"
    if [ ! -f $correctness_file ]; then
        python get_triviaqa_IK_training_set.py ${dataset} test $model_gen_file
    fi
    # python get_triviaqa_IK_training_set.py ${dataset} test $model_gen_file

    # 3. estimate the confidence of vanilla greedy search answers
    echo "estimating the confidence of vanilla greedy search answers"
    confidence_file="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_20_answers.greedy_search.IK_set.confidence_seq_avg.csv"
    if [ ! -f $confidence_file ]; then
        python gen_conf_collect_logits_for_ICL_w_answer_confidence_4_answer_normal_demos_avg.py $model_name $dataset $correctness_file $model_gen_file
    fi
    # python gen_conf_collect_logits_for_ICL_w_answer_confidence_4_answer_normal_demos_avg.py $model_name $dataset $correctness_file $model_gen_file
    
    # 4. guided greedy search
    guided_model_gen_file="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_20_answers.greedy_search.fact_guided_seq_avg.csv"
    echo "guided greedy search"
    # if [ ! -f $guided_model_gen_file ]; then
    #     python gen_model_answers_fact_guided.py $model_name $dataset $num_samples test
    # fi
    python gen_model_answers_fact_guided.py $model_name $dataset $num_samples test

    # 5. estimate the correctness of guided greedy search answers
    echo "estimating the correctness of guided greedy search answers"
    guided_correctness_file="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_20_answers.greedy_search.fact_guided_seq_avg.IK_set.csv"
    # if [ ! -f $guided_correctness_file ]; then
    #     python get_triviaqa_IK_training_set.py ${dataset} test $guided_model_gen_file
    # fi
    python get_triviaqa_IK_training_set.py ${dataset} test $guided_model_gen_file
    
    # 6. estimate the confidence of guided greedy search answers
    echo "estimating the confidence of guided greedy search answers"
    guided_confidence_file="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_20_answers.greedy_search.fact_guided_seq_avg.IK_set.confidence_seq_avg.csv"
    # if [ ! -f $guided_confidence_file ]; then
    #     python gen_conf_collect_logits_for_ICL_w_answer_confidence_4_answer_normal_demos_avg.py $model_name $dataset $guided_correctness_file $guided_model_gen_file
    # fi
    python gen_conf_collect_logits_for_ICL_w_answer_confidence_4_answer_normal_demos_avg.py $model_name $dataset $guided_correctness_file $guided_model_gen_file

    # 7. get the final results
    echo "get the final results"
    python filter_generation.py $confidence_file $guided_confidence_file "../${dataset}/test.txt"
done