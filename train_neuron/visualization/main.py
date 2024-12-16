import matplotlib.pyplot as plt
import numpy as np


def load_dev_loss_from_log_and_plot(path_list):
    for path in path_list:

        f = open(path, 'r', encoding='utf-8')

        lines = f.readlines()

        loss_list = []

        step_length = -1
        flag = False

        for line in lines:
            if len(loss_list) == 1 and step_length == -1:
                if not flag:
                    flag = True
                    continue
                step_length = int(line.split('|')[2].split('/')[0].strip())
                print(step_length)

            if 'eval_loss' in line:
                line = line.split(',')
                for l in line:
                    if 'eval_loss' in l:
                        if '=' in l:
                            continue
                        loss = l.split(':')[1].strip()
                        loss = float(loss)
                        loss_list.append(loss)
                        break

        loss_list_length = len(loss_list)
        step_list = [step_length * i for i in range(1, loss_list_length + 1)]

        if 'lora' in path:
            plt.plot(step_list, loss_list, '--', alpha=0.5, linewidth=1)
        elif 'neuron.out' in path:
            plt.plot(step_list, loss_list, '-', alpha=0.5, linewidth=1)
        elif 'org' in path:
            print('org')
            plt.plot(step_list, loss_list, ':', alpha=0.5, linewidth=1)

        min_ = min(loss_list)

        # for a, b in zip(step_list, loss_list):
        #     if b == min_:
        #         # plt.plot(a, b, 'o', alpha=1, linewidth=1)
        #         plt.text(a, b, str(b), ha='left', va='top', fontsize=8)

    plt.xticks(np.arange(0,10400, 800))
    plt.xlabel('Step')
    plt.ylabel('Eval Loss')
    plt.legend(['Full parameter training', 'LORA(in | out; rank 8)', 'LORA(in | out; rank 16)',
                'LORA(in | out; rank 64)', 'LORA(all linear; rank 8)', 'LORA(all linear; rank 16)',
                'LORA(all linear; rank 64)', 'Neuron-Level SFT(cos-6250)', 'Neuron-Level SFT(cos-2.5w)',
                'Neuron-Level SFT(cos-5w)'])
    plt.savefig('MT_enzh_compare_with_lora_all.png', dpi=600)
    plt.show()


def load_train_loss_from_log_and_plot(path):
    f = open(path, 'r', encoding='utf-8')

    lines = f.readlines()

    loss_list = []

    step_length = 10

    for line in lines:
        if "'loss'" in line:
            line = line.split(',')
            for l in line:
                if "'loss'" in l:
                    loss = l.split(':')[1].strip()
                    loss = float(loss)
                    loss_list.append(loss)
                    break

    loss_list_length = len(loss_list)
    step_list = [step_length * i for i in range(1, loss_list_length + 1)]

    plt.plot(step_list, loss_list, 'b.- ', alpha=1, linewidth=1)
    # plt.xticks(np.arange(0, step_list[-1] + step_length, 10))
    min_ = min(loss_list)

    # for a, b in zip(step_list, loss_list):
    #     if b == min_:
    #         plt.text(a, b, str(b), ha='right', va='top', fontsize=8)

    plt.xlabel('Step')
    plt.ylabel('Train Loss')

    plt.show()


if __name__ == '__main__':
    path_list = [
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/train_neuron/visualization/logs/ACL_train_enzh_org.out',
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/train_neuron/visualization/logs/ACL_mt_enzh_train_lora_rank_8_in_out.out',
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/train_neuron/visualization/logs/ACL_mt_enzh_train_lora_rank_16_in_out.out',
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/train_neuron/visualization/logs/ACL_mt_enzh_train_lora_rank_64_in_out.out',
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/train_neuron/visualization/logs/ACL_mt_enzh_train_lora_rank_8.out',
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/train_neuron/visualization/logs/ACL_mt_enzh_train_lora_rank_16.out',
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/train_neuron/visualization/logs/ACL_mt_enzh_train_lora_rank_64.out',
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/train_neuron/visualization/logs/ACL_mt_enzh_train_3200step_6250_neuron.out',
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/train_neuron/visualization/logs/ACL_mt_enzh_train_3200step_25000_neuron.out',
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/train_neuron/visualization/logs/ACL_train_enzh_3200step_50000_neuron.out'
    ]

    load_dev_loss_from_log_and_plot(path_list)
    # load_train_loss_from_log_and_plot(path)
