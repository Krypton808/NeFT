from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig, LlamaTokenizer, \
    LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
        '/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
        model_max_length=512,
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True
    )

tokenizer.pad_token = tokenizer.eos_token


test_data = """<s> Translate the following text from English to Chinese. On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.<s> 本周，斯坦福大学医学院的科学家们宣细胞类型分类的可打印的规模微型芯片，可以用标准印刷机制造，每个芯带只需要花费约1美分。</s>"""



test_inputs = tokenizer(test_data, return_tensors='pt', padding=True, max_length=True)
print(test_inputs)
print(len(test_inputs['input_ids'][0]))