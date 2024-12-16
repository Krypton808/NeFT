from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig


def test1():
    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"
    prompt_input = tok_ins + "{instruction}" + tok_res

    prompt = 'Translate the following text from English to Chinese.'
    input = "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."

    instruction = prompt + '\n' + input

    input_text = prompt_input.format_map({'instruction': instruction})
    print(input_text)

    # 71 153
    # raw_input_text = """<s>
    #
    # ### Instruction:
    # Translate the following English statements to Chinese.
    # On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.
    #
    # ### Response:
    # """

    tokenizer = AutoTokenizer.from_pretrained(
        '/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer([input_text], return_tensors='pt', padding=True)
    print(inputs)

    print(len(inputs['input_ids'][0]))

    print(tokenizer.decode([29901], skip_special_tokens=False, spaces_between_special_tokens=False))


def test2():
    f = open('/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False_2/1/output.txt')
    input_text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(
        '/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer([input_text, "hello world!"], return_tensors='pt', padding=True)
    input_ids = inputs['input_ids'][0][1:]

    print(input_ids)
    print(tokenizer.decode(input_ids))

    print(len(input_ids))


def test3():
    tokenizer = AutoTokenizer.from_pretrained(
        '/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True
    )

    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"
    prompt_input = tok_ins + "{instruction}" + tok_res

    prompt = "Translate the following text from English to Chinese."
    text = "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."

    output_s = "在星期一，stanford大学医学学院的科学家宣布了一种新的诊断工具，可以根据细胞类型排序：一个可以使用普通喷墨打印机制造的小型可读取芯片，每个芯平可能只需要大约一美元。"

    input_s = prompt + '\n' + text

    input_s = prompt_input.format_map({'instruction': input_s})

    # tokenized = tokenizer([input_s, output_s])
    tokenized = tokenizer([input_s+output_s])
    token_ids = [tok_id for tok_ids in tokenized['input_ids'] for tok_id in tok_ids]
    attention_mask = [mask for masks in tokenized['attention_mask'] for mask in masks]
    if token_ids[-1] != tokenizer.eos_token_id:
        token_ids += [tokenizer.eos_token_id]
        attention_mask += [1]
    processed_record = {
        "input_ids": token_ids,
        "attention_mask": attention_mask,
        "labels": token_ids.copy()
    }

    input_ids = processed_record['input_ids']

    print(input_ids)
    print(len(input_ids))
    print(tokenizer.decode(input_ids))



if __name__ == '__main__':
    test3()