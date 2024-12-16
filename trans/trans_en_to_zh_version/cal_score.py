import re


def gather_output_into_one_file_zh(path_dir=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False_2/'):
    path_w = path_dir + r'output_all.txt'
    w = open(path_w, 'w', encoding='utf-8')

    output_filename = 'output.txt'
    for i in range(1, 998):
        if i % 50 == 0:
            print(i)
        list_dir = path_dir + str(i)
        output_file = list_dir + '/' + output_filename
        output_text_list = []
        f = open(output_file, 'r', encoding='utf-8')
        text = f.read()
        if text == '':
            continue

        print('**********************************')
        print(text)
        text = text.split('### Response:\n')

        text = text[1].strip()
        findall = re.findall(
            r'(?:.*(?:在中文|中文).*(?:翻译|翻譯).*：|.*(?:翻译|翻譯).*中文.*：|對于以下文本的翻译：|这个文本将被翻译成中文。|我很高兴能够帮助您翻译文本。|翻譯結果：|將以下文本翻譯為中文。|中文：|(?:示例|翻譯|翻译)：|從英文轉中文：|將英文文本轉換為中文，請提供以下內容：|Translate:)',
            text)
        if findall != []:
            text = text.replace(findall[0], '').strip()

        text = text.split('\n')[0]
        text = text.split('</s>')[0].strip()
        text = text.replace('<s>', '').strip()
        print(text)
        print('*************************************')

        w.write(text + '\n')

def gather_output_into_one_file_de(path_dir=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_de_use_cache_False/'):
    path_w = path_dir + r'output_all.txt'
    w = open(path_w, 'w', encoding='utf-8')

    output_filename = 'output.txt'
    for i in range(1, 998):
        if i % 50 == 0:
            print(i)
        list_dir = path_dir + str(i)
        output_file = list_dir + '/' + output_filename
        output_text_list = []
        f = open(output_file, 'r', encoding='utf-8')
        text = f.read()
        if text == '':
            continue

        print('**********************************')
        print(text)
        text = text.split('### Response:\n')

        text = text[1].strip()
        findall = re.findall(r'Translation:|In short:|Übersetzen Sie das folgende Text aus Englisch nach Deutsch.|Zum Übersetzen ins Deutsche:|Deutsch:|In Deutsch:|In den folgenden Texten wird der Text aus Englisch nach Deutsch übersetzt:|Natürlich, ich kann Ihnen helfen, das Text von Englisch nach Deutsch zu übersetzen. Hier ist die übersetzte Version:|.*Deutsche:|English to German:|German Translation:|Für die Anweisungen:', text)
        if findall != []:
            text = text.replace(findall[0], '').strip()

        text = text.split('\n')[0]
        text = text.split('</s>')[0].strip()
        text = text.replace('<s>', '').strip()
        print(text)
        print('*************************************')

        w.write(text + '\n')



if __name__ == '__main__':
    # gather_output_into_one_file_zh()
    # gather_output_into_one_file_zh(path_dir=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_qlora_3w_3epoch_use_cache_False/')
    gather_output_into_one_file_de(path_dir=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_de_qlora_3w_3epoch_use_cache_False/')


