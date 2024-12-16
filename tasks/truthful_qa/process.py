import pandas as pd
import jsonlines
import csv
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer


def process_truthful_qa():
    datapd = pd.read_parquet('/data5/haoyun.xu/data/truthful_qa/validation-00000-of-00001.parquet')
    writer = jsonlines.open('/data5/haoyun.xu/data/truthful_qa/truthful_qa.jsonl', mode='w')

    prefix = "[INST]Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"

    for data in datapd.iterrows():
        data = data[1]
        question = prefix + data['question'] + "[/INST]"
        best_answer = data['best_answer']
        correct_answers = data['correct_answers']
        print(correct_answers)
        incorrect_answers = data['incorrect_answers']
        temp_dict = {}
        temp_dict['instruction'] = question
        temp_dict['correct_answers'] = [i for i in correct_answers]
        temp_dict['incorrect_answers'] = [i for i in incorrect_answers]
        writer.write(temp_dict)


def eval_truthful_qa_Bleu(
        output_path="/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/tasks/truthful_qa/output/truthful_qa_NeFT_150000_39000_2.csv",
        ref_path='/data5/haoyun.xu/data/truthful_qa/truthful_qa.jsonl'):
    bleu = BLEU(trg_lang='en')
    csvfile = open(output_path, 'r', encoding='utf-8')
    rows = csv.reader(csvfile)
    sys_list = []
    for row in rows:
        sys = row[0].replace("###, Response:", "").replace("##|", "").replace("###,", "").replace("###", "").replace(
            "##", "").strip()
        sys_list.append(sys)

    lines = jsonlines.open(ref_path, mode='r')

    final_score_sum = 0
    final_score_list = []
    for idx, line in enumerate(zip(lines, sys_list)):
        correct_answers = line[0]['correct_answers']
        incorrect_answers = line[0]['incorrect_answers']
        corr_max = -1
        incorr_max = -1

        refs = [line[1]]
        for corr in correct_answers:
            score = bleu.sentence_score(corr, refs).score
            if corr_max < score:
                corr_max = score

        for incorr in incorrect_answers:
            score = bleu.sentence_score(incorr, refs).score
            if incorr_max < score:
                incorr_max = score

        score_final = corr_max - incorr_max
        final_score_list.append(score_final)
        final_score_sum += score_final

    score_f = final_score_sum / (idx + 1)
    print(score_f)


def eval_truthful_qa_rouge(output_path="/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/tasks/truthful_qa/output/truthful_qa_base_full_39000.csv",
        ref_path='/data5/haoyun.xu/data/truthful_qa/truthful_qa.jsonl'):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    csvfile = open(output_path, 'r', encoding='utf-8')
    rows = csv.reader(csvfile)
    sys_list = []
    for row in rows:
        sys = row[0].replace("###, Response:", "").replace("##|", "").replace("###,", "").replace("###", "").replace(
            "##", "").strip()
        sys_list.append(sys)

    lines = jsonlines.open(ref_path, mode='r')

    final_score_sum = 0
    final_score_list = []
    for idx, line in enumerate(zip(lines, sys_list)):
        correct_answers = line[0]['correct_answers']
        incorrect_answers = line[0]['incorrect_answers']
        corr_max = -1
        incorr_max = -1

        for corr in correct_answers:
            score = rouge.score(corr, line[1])['rougeL'].fmeasure
            if corr_max < score:
                corr_max = score

        for incorr in incorrect_answers:
            score = rouge.score(incorr, line[1])['rougeL'].fmeasure
            if incorr_max < score:
                incorr_max = score

        score_final = corr_max - incorr_max
        final_score_list.append(score_final)
        final_score_sum += score_final

    score_f = final_score_sum / (idx + 1)
    print(score_f)

    # scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    # scores = scorer.score("hello world", "hello world")
    # print(scores['rougeL'].fmeasure)



if __name__ == '__main__':
    eval_truthful_qa_rouge()
