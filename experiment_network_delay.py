from main import main
import numpy as np
from pathlib import Path
import time
import json
from collections import defaultdict
from executor import Executor
from multiprocessing import Process, Queue

def single_process(result_path, miner_num, blocksize, task_type, queue:Queue):
    total_rounds = 500000
    height = 200
    #matrix = np.ones([miner_num,miner_num]) - np.eye(miner_num)
    #network_topo_path = Path().cwd()/"network"/"topologies"
    #matrix = np.loadtxt(network_topo_path/f"miner_num_{miner_num}.csv",delimiter=',')
    result=main(total_rounds, n=miner_num, blocksize = blocksize,
                result_path=result_path/f"blocksize_{blocksize}",
                max_height=height, network_generator='coo',
                task_selection=task_type)
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
    # 待测参数序列
    TASK_TYPE = 'A'
    BLOCKSIZE = [32,24,16,12,8,6,4,2] # [16,8,6,4,2]
    MINER_NUM = [10]
    if TASK_TYPE.endswith('DTC') or TASK_TYPE == 'A':
        POOL_SIZE = 8 # Constrained by CPU
    else:
        POOL_SIZE = 1 # Constrained by GPU
    print('BLOCKSIZE =', BLOCKSIZE, file=result_file)
    print('MINER_NUM =', MINER_NUM, file=result_file)
    print('test_score_mainchain = {}', file=result_file)
    print('validation_score_mainchain = {}', file=result_file)
    print('test_score_upperbound = {}', file=result_file)
    print('validation_score_upperbound = {}', file=result_file)
    print('average_generated_miniblocks = {}', file=result_file)
    print('average_valid_miniblocks = {}', file=result_file)

    result_collection = defaultdict(list)

    executor = Executor(available_cpu_id=list(range(0,POOL_SIZE * 2,2)))

    for miner_num in MINER_NUM:
        print('miner_num =', miner_num, file=result_file)
        test_scores = []
        validation_scores = []
        average_generated_miniblocks = []
        average_valid_miniblocks = []
        test_score_upperbound = []
        validation_score_upperbound = []
        # 模拟Apply函数式编程
        queue_list = [Queue() for _ in BLOCKSIZE]
        param_array = [(result_path, miner_num, blocksize, TASK_TYPE, queue) for blocksize,queue in zip(BLOCKSIZE, queue_list)]
        result_fallback_list = [result_fallback_function(result_path/f"blocksize_{blocksize}") for blocksize in BLOCKSIZE]
        process_list = [Process(target=single_process, args=params) for params in param_array]
        results = executor.par_run(process_list, queue_list, process_file, result_fallback_list)

        for result in results:
            test_scores.append(result['test_metric_average'])
            validation_scores.append(result['validation_metric_average'])
            average_generated_miniblocks.append(result['average_generated_miniblocks_per_height'])
            average_valid_miniblocks.append(result['average_valid_miniblocks_per_height'])
            test_score_upperbound.append(result['average_accuracy_upper_bound_for_all_generated_miniblocks'][0])
            validation_score_upperbound.append(result['average_accuracy_upper_bound_for_all_generated_miniblocks'][1])
            result_collection[miner_num].append(result)

        print(f'test_score_mainchain[{miner_num}]=', test_scores, file=result_file)
        print(f'validation_score_mainchain[{miner_num}]=', validation_scores, file=result_file)
        print(f'test_score_upperbound[{miner_num}]=', test_score_upperbound, file=result_file)
        print(f'validation_score_upperbound[{miner_num}]=', validation_score_upperbound, file=result_file)
        print(f'average_generated_miniblocks[{miner_num}]=', average_generated_miniblocks, file=result_file)
        print(f'average_valid_miniblocks[{miner_num}]=', average_valid_miniblocks, file=result_file)

    result_file.close()
    process_file.close()

    import json
    with open(result_path / 'result_archive.json', 'w') as f:
        # 模拟Apply函数式编程
        #[(lambda x: x['chain_stats'].update({'common_prefix_pdf': x['chain_stats']['common_prefix_pdf'].tolist()}))(result) for param in result_collection for result in result_collection[param]]
        json.dump(result_collection, f)