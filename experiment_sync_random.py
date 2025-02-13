from main import main
import numpy as np
from pathlib import Path
import time
from collections import defaultdict

# import sys
# import os
# print(sys.path)
# print(os.environ['PATH'])

def single_process(result_path:Path, miner_num, blocksize, task_type, noniid_conf, queue, seq:int, net_type:str):
    total_rounds = 50000
    height = 15
    # Originally, miner_num == partition_num for different miner_num
    # noniid_conf['partition_num'] = miner_num
    if net_type == 'sync':
        matrix = np.ones([miner_num,miner_num]) - np.eye(miner_num)
    else: # 'random'
        network_topo_path = Path().cwd()/"network"/"topologies"/f"collection{seq}"
        matrix = np.loadtxt(network_topo_path/f"miner_num_{miner_num}.csv",delimiter=',')
    
    (result_path / net_type / f"{seq}").mkdir(parents=True, exist_ok=True)
    #network_topo_path = Path().cwd()/"network"/"topologies"
    #matrix = np.loadtxt(network_topo_path/f"miner_num_{miner_num}.csv",delimiter=',')
    result=main(total_rounds, n=miner_num, blocksize = blocksize,
                result_path=result_path/net_type/f"{seq}"/f"miner_num_{miner_num}",
                max_height=height, network_generator='matrix',
                matrix=matrix, task_selection=task_type, noniid_conf=noniid_conf)
    queue.put(result)
    return result

def result_fallback_function(result_path):
    def result_fallback():
        with open(result_path / 'result_collection.json', 'r') as f:
            return json.load(f)
    return result_fallback

if __name__ == '__main__':
    # here the global ratio is the ratio of global sample counts over the total sample count at each node
    current_time = time.strftime("%Y%m%d-%H%M%S")
    result_path=Path.cwd() / 'Results' / current_time
    result_path.mkdir(parents=True)
    result_file = open(result_path / 'result.txt', 'w')
    process_file = open(result_path / 'process.log', 'w')
    noniid_conf = {'type':'label_quantity', 'global_ratio':0.4, 'label_per_miner':10, 'beta':0.5}
    # label_per_miner = 10 for CIFAR10 to ensure IID label distribution
    # 待测参数序列
    TASK_TYPE = 'C-CIFAR10-ResNet18'
    #GLOBAL_TRAINING_SET_RATIO = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    #BETA = [0.1, 0.25, 0.5, 1, 5, 15] # [5, 15]
    REPETITION = 1 # 6
    MINER_NUM = [25, 20, 15] # [3, 25, 7, 10, 20, 15, 5, 12] # [3, 15, 12, 7, 10, 5] [3, 5, 7, 10, 12, 15]
    noniid_conf['partition_num'] = 10 # 15 #if 15 > max(MINER_NUM) else max(MINER_NUM)
    BLOCKSIZE = 6
    if TASK_TYPE.endswith('DTC') or TASK_TYPE == 'A':
        POOL_SIZE = 8 # Constrained by CPU
    else:
        POOL_SIZE = 1 # Constrained by GPU
    #print('GLOBAL_TRAINING_SET_RATIO =', GLOBAL_TRAINING_SET_RATIO, file=result_file)
    #print('BETA =', BETA, file=result_file)
    print('MINER_NUM =', MINER_NUM, file=result_file)
    print('PR =', noniid_conf['global_ratio'], file=result_file)
    print('partition_num =', noniid_conf['partition_num'], file=result_file)
    print('test_set_metric = defaultdict(list)', file=result_file)
    print('validation_set_metric = defaultdict(list)', file=result_file)
    print('test_set_metric_bad_network = defaultdict(list)', file=result_file)
    print('validation_set_metric_bad_network = defaultdict(list)', file=result_file)
    result_collection = defaultdict(list)

    from executor import Executor
    from multiprocessing import Process, Queue
    executor = Executor(available_cpu_id=list(range(0, POOL_SIZE*2, 2)))

    for net_type in ['sync']: # ['sync', 'random']
        START = 6
        for seq in range(START, START+REPETITION):
            #print('#GLOBAL_RATIO =', global_ratio, file=result_file)
            test_scores = []
            validation_scores = []
            miner_num = MINER_NUM if net_type == 'sync' else MINER_NUM[1:] # sync network = random network when miner_num=3
            queue_list = [Queue() for _ in miner_num]
            param_array = [(result_path, num, BLOCKSIZE, TASK_TYPE, noniid_conf, queue, seq, net_type) for num,queue in zip(miner_num, queue_list)]
            result_fallback_list = [result_fallback_function(result_path/net_type/f"{seq}"/f"miner_num_{num}") for num in miner_num]
            process_list = [Process(target=single_process, args=params) for params in param_array]
            results = executor.par_run(process_list, queue_list, process_file)

            for result in results:
                test_scores.append(result['test_metric_average'])
                validation_scores.append(result['validation_metric_average'])
                result_collection[seq].append(result)
            if net_type == 'sync':
                print(f'test_set_metric[partition_num].append(np.array(', test_scores, '))', file=result_file)
                print(f'validation_set_metric[partition_num].append(np.array(', validation_scores, '))',
                      file=result_file, flush=True)
            else:
                print(f'test_set_metric_bad_network[partition_num].append(np.array(', test_scores, '))', file=result_file)
                print(f'validation_set_metric_bad_network[partition_num].append(np.array(', validation_scores, '))',
                      file=result_file, flush=True)

        if net_type == 'sync':
            print('test_set_metric[partition_num] = np.vstack(test_set_metric[partition_num])', file=result_file)
            print('validation_set_metric[partition_num] = np.vstack(validation_set_metric[partition_num])',
                  file=result_file, flush=True)
        else:
            print('test_set_metric_bad_network[partition_num] = np.vstack(test_set_metric_bad_network[partition_num])', file=result_file)
            print('validation_set_metric_bad_network[partition_num] = np.vstack(validation_set_metric_bad_network[partition_num])',
                  file=result_file, flush=True)

    result_file.close()
    process_file.close()

    import json
    with open(result_path / 'result_archive.json', 'w') as f:
        # 模拟Apply函数式编程
        #[(lambda x: x['chain_stats'].update({'common_prefix_pdf': x['chain_stats']['common_prefix_pdf'].tolist()}))(result) for param in result_collection for result in result_collection[param]]
        json.dump(result_collection, f)
