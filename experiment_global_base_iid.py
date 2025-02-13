from main import main
import numpy as np
from pathlib import Path
import time
from collections import defaultdict

EPOCH = 20

def single_process(result_path, miner_num, blocksize, task_type, noniid_conf, queue):
    total_rounds = 50000
    height = 15
    matrix = np.ones([miner_num,miner_num]) - np.eye(miner_num)
    #network_topo_path = Path().cwd()/"network"/"topologies"
    #matrix = np.loadtxt(network_topo_path/f"miner_num_{miner_num}.csv",delimiter=',')
    result=main(total_rounds, n=miner_num, blocksize = blocksize,
                result_path=result_path/f"gr{noniid_conf['global_ratio']}"/f"beta{noniid_conf['beta']}",
                max_height=height, network_generator='matrix',
                matrix=matrix, task_selection=task_type, noniid_conf=noniid_conf,
                epoch=EPOCH)
    queue.put(result)
    return result

if __name__ == '__main__':
    # here the global ratio is the ratio of global sample counts over the total sample count at each node
    current_time = time.strftime("%Y%m%d-%H%M%S")
    result_path=Path.cwd() / 'Results' / current_time
    result_path.mkdir(parents=True)
    result_file = open(result_path / 'result.txt', 'w')
    process_file = open(result_path / 'process.log', 'w')
    # 待测参数序列
    TASK_TYPE = 'C-SVHN-ResNet18' # C-SVHN-ResNet18
    #GLOBAL_TRAINING_SET_RATIO = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    GLOBAL_TRAINING_SET_RATIO = [1, 0.4, 0.3, 0.08, 0.02, 0.005]
    BETA = [5] # [0.25, 0.5, 1, 5]
    MINER_NUM = 10
    BLOCKSIZE = 6
    noniid_conf = {'type':'label_quantity', 'global_ratio':0.1,
                   'label_per_miner':MINER_NUM, 'base_global_experiment': True}
    if TASK_TYPE.endswith('DTC') or TASK_TYPE == 'A':
        POOL_SIZE = 8 # Constrained by CPU
    else:
        POOL_SIZE = 2 # Constrained by GPU
    print('GLOBAL_TRAINING_SET_RATIO =', GLOBAL_TRAINING_SET_RATIO, file=result_file)
    print('BETA =', BETA, file=result_file)
    print('test_score_beta = {}', file=result_file)
    print('test_score_base_beta = {}', file=result_file)
    result_collection = defaultdict(list)

    from executor import Executor
    from multiprocessing import Process, Queue
    executor = Executor(available_cpu_id=list(range(POOL_SIZE)))

    test_scores_base_global = [] # 全局模型
    validation_scores_base_global = [] # 全局模型
    for beta in BETA:
        print('#BETA =', beta, file=result_file)
        test_scores = [] # 集成模型
        validation_scores = []
        test_scores_base = [] # 基模型
        validation_scores_base = []
        test_scores_global_list = [] # 全局模型
        validation_scores_global_list = []
        # 模拟Apply函数式编程
        conf_list = [(lambda x: x.update({'global_ratio': global_ratio, 'beta': beta}) or x)(noniid_conf.copy()) for global_ratio in GLOBAL_TRAINING_SET_RATIO]
        queue_list = [Queue() for _ in conf_list]
        param_array = [(result_path, MINER_NUM, BLOCKSIZE, TASK_TYPE, conf, queue) for conf,queue in zip(conf_list, queue_list)]
        process_list = [Process(target=single_process, args=params) for params in param_array]
        results = executor.par_run(process_list, queue_list, process_file)

        for result in results:
            test_scores.append(result['test_metric_average'])
            validation_scores.append(result['validation_metric_average'])
            test_scores_base.append(result['base_model_metric_test'])
            validation_scores_base.append(result['base_model_metric_validation'])
            test_scores_global_list.append(result['test_metric_on_global_dataset'])
            validation_scores_global_list.append(result['validation_metric_on_global_dataset'])
            result_collection[beta].append(result)
        test_scores_base_global.append(test_scores_global_list)
        validation_scores_base_global.append(validation_scores_global_list)
        # 暂时只使用测试集的指标
        print(f'test_score_beta[{beta}]=', test_scores, file=result_file)
        print(f'test_score_base_beta[{beta}]=',
              'np.array(' + str(test_scores_base).replace('], [','],\n[') + ').T', file=result_file)
    
    test_scores_base_global = np.array(test_scores_base_global).mean(axis=0).tolist()
    validation_scores_base_global = np.array(validation_scores_base_global).mean(axis=0).tolist()
    print('test_score_global=', test_scores_base_global, file=result_file)

    result_file.close()
    process_file.close()

    import json
    with open(result_path / 'result_archive.json', 'w') as f:
        # 模拟Apply函数式编程
        [(lambda x: x['chain_stats'].update({'common_prefix_pdf': x['chain_stats']['common_prefix_pdf'].tolist()}))(result) for param in result_collection for result in result_collection[param]]
        json.dump(result_collection, f)