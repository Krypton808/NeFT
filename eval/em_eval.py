def em(path_1='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/other/data/ref.txt',
       path_2='/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/other/lora/6250step/output.csv'):

    f = open(path_1, 'r', encoding='utf-8')
    f_2 = open(path_2, 'r', encoding='utf-8')
    lines = f.readlines()
    lines_2 = f_2.readlines()
    length = len(lines)
    count = 0

    for line in zip(lines, lines_2):
        if line[0].strip() in line[1]:
            count += 1


    print(count / length)


if __name__ == '__main__':
    em()
