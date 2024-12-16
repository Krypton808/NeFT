import csv
import re

def clean_whether_chinese():
    file_obj = open("/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/summary/llama2_chat/enzh_output_2.csv", 'w', encoding='utf-8')
    writer = csv.writer(file_obj)

    with open("/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/summary/llama2_chat/enzh_output_.csv",
              'r', encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            reg = re.findall('[^\x00-\xff]', row[0])
            if reg == []:
                row = [""]

            print(row)
            print(type(row))
            writer.writerow(row)


def read_csv():
    with open("/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/summary/llama2_chat/enzh_output_2.csv",
              'r', encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            print(row)

if __name__ == "__main__":
    clean_whether_chinese()
    read_csv()