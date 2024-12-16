import datasets
import jsonlines

def old():
    datasets = datasets.load_dataset('/mnt/nfs/algo/intern/haoyunx9/data/xquad')
    print(datasets)

    prefix_1 = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGiven the following question, find and output the corresponding answer from the given passage\n\n### Input:\nQuestion: '
    prefix_2 = '\nPassage: '
    subfix = '\n\n'

    # write = jsonlines.open('/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/other/data/data.jsonl', 'w')
    # for data in datasets['train']:
    #     text = data['text'].split("### Response:\n")
    #     input = text[0]
    #     output = '### Response:\n' + text[1]
    #
    #     # print(input)
    #     # print(output)
    #
    #     temp_dict = {}
    #     temp_dict['input'] = input
    #     temp_dict['output'] = output
    #
    #     write.write(temp_dict)
    #     print('------------')

    w = open('/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/other/data/ref.txt', 'w', encoding='utf-8')
    # write = jsonlines.open('/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/other/data/test.jsonl', 'w')
    for data in datasets['validation']:
        context = data['context']
        question = data['question']
        output = data['answers']['text'][0]
        if len(data['answers']['text']) != 1:
            print('LLLLL')
        input = prefix_1 + question + prefix_2 + context + subfix

        w.write(output + '\n')


        # print(input)
        # print(output)

        temp_dict = {}
        temp_dict['input'] = input
        temp_dict['output'] = output

        # write.write(temp_dict)
        # print('------------')

def process_alpaca():
    data_path = "/mnt/nfs/algo/intern/haoyunx9/data/alpaca"
    lines = datasets.load_dataset(data_path)
    writer = jsonlines.open("/data5/haoyun.xu/data/alpaca/alpaca.jsonl", mode='w')

    for line in lines['train']:
        temp_dict = {}
        text = line['text'].split('\n\n### Response:')

        temp_dict['instruction'] = '[INST]' + text[0] + '[/INST]'
        temp_dict['output'] = '\n\n### Response:' + text[1]

        writer.write(temp_dict)


if __name__ == '__main__':
    process_alpaca()

