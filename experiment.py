import time
from pathlib import Path
from multiprocessing import Pool

from main import main

def single_process(result_path,miner_num):    
    total_rounds = 150 # 510
    main(total_rounds, miner_num, blocksize=2,
        result_path=result_path/f"miner_num_{miner_num}")

if __name__ == "__main__":
    current_time = time.strftime("%Y%m%d-%H%M%S")
    result_path=Path.cwd() / 'Results' / current_time
    # miner_num_experiment_list = [5,7,10,12,15]
    miner_num_experiment_list = [5,6,7]
    # for miner_num in miner_num_experiment_list:
        # single_process(miner_num)
    pool_size = 4 # CPU Core Number
    with Pool(pool_size) as p:
        for miner_num in miner_num_experiment_list:
            p.apply_async(single_process,args=(result_path,miner_num))
        p.close()
        p.join()
