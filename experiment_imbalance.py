from main import main
import numpy as np
from pathlib import Path
import time
from collections import defaultdict
def single_process(result_path, miner_num, blocksize, task_type, noniid_conf, queue):
    total_rounds = 50000
    height = 15
    matrix = np.ones([miner_num,miner_num]) - np.eye(miner_num)
    #network_topo_path = Path().cwd()/"network"/"topologies"
    #matrix = np.loadtxt(network_topo_path/f"miner_num_{miner_num}.csv",delimiter=',')
    if noniid_conf['type'] == 'label_quantity':
        pathname = f"label_per_miner{noniid_conf['label_per_miner']}"
    else:
        pathname = f"beta{noniid_conf['beta']}"
    result=main(total_rounds, n=miner_num, blocksize = blocksize,
                result_path=result_path/f"gr{noniid_conf['global_ratio']}"/pathname,
                max_height=height, network_generator='matrix',
                matrix=matrix, task_selection=task_type, noniid_conf=noniid_conf)
    queue.put(result)
    return result

if __name__ == '__main__':
    # here the global ratio is the ratio of global sample counts over the total sample count at each node
    current_time = time.strftime("%Y%m%d-%H%M%S")
    result_path=Path.cwd() / 'Results' / current_time
    result_path.mkdir(parents=True)
    result_file = open(result_path / 'result.txt', 'w')
    process_file = open(result_path / 'process.log', 'w')
    noniid_conf = {'type':'label_distribution', 'global_ratio':0.1, 'label_per_miner':3, 'beta':0.5}
    # 待测参数序列
    TASK_TYPE = 'C-CIFAR10-ResNet18' # 'C-SVHN-ResNet18' # 
    GLOBAL_TRAINING_SET_RATIO =  [0.2, 0.1, 0.02] # [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    BETA = [0.1, 0.25, 0.5, 1, 5, 15]
    LABEL_PER_MINER = [2,3,5,7,9,10]
    MINER_NUM = 10
    BLOCKSIZE = 6
    if TASK_TYPE.endswith('DTC') or TASK_TYPE == 'A':
        POOL_SIZE = 8 # Constrained by CPU
    else:
        POOL_SIZE = 2 # Constrained by GPU
    print('GLOBAL_TRAINING_SET_RATIO =', GLOBAL_TRAINING_SET_RATIO, file=result_file)
    if noniid_conf['type'] == 'label_quantity':
        print('LABEL_PER_MINER =', LABEL_PER_MINER, file=result_file)
        print('test_score_label_quantity = {}', file=result_file)
        print('validation_score_label_quantity = {}', file=result_file)
    else:
        print('BETA =', BETA, file=result_file)
        print('test_score_label_distr = {}', file=result_file)
        print('validation_score_label_distr = {}', file=result_file)

    result_collection = defaultdict(list)

    from executor import Executor
    from multiprocessing import Process, Queue
    executor = Executor(available_cpu_id=list(range(POOL_SIZE)))

    for global_ratio in GLOBAL_TRAINING_SET_RATIO:
        print('#GLOBAL_RATIO =', global_ratio, file=result_file)
        test_scores = []
        validation_scores = []
        # 模拟Apply函数式编程
        if noniid_conf['type'] == 'label_quantity':
            conf_list = [(lambda x: x.update({'global_ratio': global_ratio, 'label_per_miner': quantity}) or x)\
                         (noniid_conf.copy()) for quantity in LABEL_PER_MINER]
        else:
            conf_list = [(lambda x: x.update({'global_ratio': global_ratio, 'beta': beta}) or x)\
                         (noniid_conf.copy()) for beta in BETA]
        queue_list = [Queue() for _ in conf_list]
        param_array = [(result_path, MINER_NUM, BLOCKSIZE, TASK_TYPE, conf, queue) for conf,queue in zip(conf_list, queue_list)]
        process_list = [Process(target=single_process, args=params) for params in param_array]
        results = executor.par_run(process_list, queue_list, process_file)

        for result in results:
            test_scores.append(result['test_metric_average'])
            validation_scores.append(result['validation_metric_average'])
            result_collection[global_ratio].append(result)

        if noniid_conf['type'] == 'label_quantity':
            print(f'test_score_label_quantity[{global_ratio}]=', test_scores, file=result_file)
            print(f'validation_score_label_quantity[{global_ratio}]=', validation_scores, file=result_file)
        else:
            print(f'test_score_label_distr[{global_ratio}]=', test_scores, file=result_file)
            print(f'validation_score_label_distr[{global_ratio}]=', validation_scores, file=result_file)

    result_file.close()
    process_file.close()

    import json
    with open(result_path / 'result_archive.json', 'w') as f:
        # 模拟Apply函数式编程
        [(lambda x: x['chain_stats'].update({'common_prefix_pdf': x['chain_stats']['common_prefix_pdf'].tolist()}))(result) for param in result_collection for result in result_collection[param]]
        json.dump(result_collection, f)
