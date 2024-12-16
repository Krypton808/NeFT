import csv
import jsonlines


# /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/tasks/mmlu/output/LoRA/rank_64_2/39000/output.csv # 0.458
# /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/tasks/mmlu/output/LoRA/rank_64_2_qkvo/output.csv # 0.441
def accuracy(
        in_path='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/tasks/mmlu/output/LoRA/rank_64_2/39000/output.csv',
        test_set_path='/data5/haoyun.xu/data/mmlu/all/test/test.jsonl',
        out_path='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/alpaca/output/39000_processed.csv'):
    output_list = []
    gt_list = []
    lines = jsonlines.open(test_set_path, mode='r')
    for line in lines:
        if line['output'] == 0:
            gt_list.append('A')
        elif line['output'] == 1:
            gt_list.append('B')
        elif line['output'] == 2:
            gt_list.append('C')
        elif line['output'] == 3:
            gt_list.append('D')

    UPPER_LETTER = ['A', 'B', 'C', 'D']

    with open(in_path, 'r', encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            # print(row)
            if 'Response:' in row[0]:
                answer = row[0].split('Response:')[1][0]
                output_list.append(answer)
            else:
                answer = 'N'
                for idx, c in enumerate(row[0]):
                    if c in UPPER_LETTER:
                        if idx < len(row[0]) - 1:
                            if row[0][idx + 1] == '.':
                                answer = c
                                break

                if answer == 'N':
                    print(row[0])

                output_list.append(answer)

    corr = 0
    for line in zip(output_list, gt_list):
        print(line[0])
        if line[0] == line[1]:
            corr += 1

    acc = corr / len(gt_list)
    print(acc)


if __name__ == '__main__':
    accuracy()
