import numpy as np

def test1():
    # # embedding_1 = np.load(f'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh/1/generated_token_2_hidden_layer_32.npy')
    # embedding_2 = np.load(f'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh/1/generated_token_2_hidden_layer_33.npy')
    # #
    # # # print(embedding_1)
    # # # print(embedding_1.shape)
    # # # print('+++++++++++++++++++++++')
    # print(embedding_2)
    # embedding_2 = embedding_2.squeeze()
    # embedding_2 = embedding_2.reshape((1, 33, 4096))
    # print('***************')
    # print(embedding_2)
    # print(embedding_2.shape)
    # # print(embedding_2[:, 0, :])
    # # print(embedding_2[0, :, :].shape)
    #
    # # embedding_en = np.load(r'/data5/haoyun.xu/study/MI/MMMI/src/example/Ouput/Embeddings/base_en.npy')
    # # embedding_zh = np.load(r'/data5/haoyun.xu/study/MI/MMMI/src/example/Ouput/Embeddings/base_zh.npy')
    # # print(embedding_zh)
    # # print(embedding_zh.shape)   # (5010, 13, 768) == (example_number, hidden_layers_num, hidden_states)

    embedding_1 = np.load(f'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_test/1/generated_token_106_hidden_layer_33.npy') # org use_cache=True
    embedding_2 = np.load(f'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False/1/generated_token_106_hidden_layer_33.npy') # org use_cache=False
    embedding_3 = np.load(f'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_qlora_3w_3epoch/1/generated_token_129_hidden_layer_33.npy') # lora use_cahce=True
    embedding_4 = np.load(f'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_qlora_3w_3epoch_use_cache_False/1/generated_token_114_hidden_layer_33.npy') # lora use_cahce=True

    # embedding_3 = np.load(f'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_qlora_3w_3epoch/1/generated_token_131_hidden_layer_33.npy') # qlora use_cache=True
    # embedding_4 = np.load(f'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh/1/generated_token_2_hidden_layer_33.npy')




    print(embedding_1.shape)
    print(embedding_2.shape)
    print(embedding_3.shape)
    print(embedding_4.shape)

    # print(embedding_1[:,:,-1,:])
    # print('+++++++++++++++++++++++')
    # print(embedding_2[:,:,-2,:])

    # print(embedding_1[:,:,-1,:])
    # print('+++++++++++++++++++++++')
    # print(embedding_3[:,:,-1,:])


def test2():
    embedding_1 = np.load(f'/data5/haoyun.xu/study/MI/MMMI/src/example/Ouput/Embeddings/base_en.npy')
    embedding_2 = np.load(f'/data5/haoyun.xu/study/MI/MMMI/src/example/Ouput/Embeddings/base_zh.npy')

    print(embedding_1.shape)
    print(embedding_2.shape)

def test3():
    embedding_1 = np.load(f'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False_2/1/1_hidden_layer_33.npy')
    embedding_2 = np.load(f'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False_2/1/110_hidden_layer_33.npy')

    print(embedding_1)
    print('*************************')
    print(embedding_2)

    print(embedding_1.shape)
    print(embedding_2.shape)

if __name__ == '__main__':
    test2()


