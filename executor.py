import multiprocessing
from multiprocessing import Process, Queue
from queue import Empty
# import psutil

def assign_cpu(pids:list[int], cpu_ids:list[int]) -> dict[int, int]:
    cpu_id_list = cpu_ids.copy()
    pid_cpu_id_mapping = {}
    for pid in pids:
        if cpu_id_list:
            pid_cpu_id_mapping[pid] = cpu_id_list.pop(0)
            # psutil.Process(pid).cpu_affinity([pid_cpu_id_mapping[pid]])
    return pid_cpu_id_mapping

class Executor:
    def __init__(self, *, available_cpu_id:list[int] = None, p_core_id:list[int] = None):
        self.available_cpu_id = available_cpu_id or [0]
        self.p_core_id = p_core_id or [0,2,4,6,8,10,12,14]
        self.e_core_id = [id for id in available_cpu_id if id not in self.p_core_id]
        self.p_core_mask_list = [1 << cpu_id for cpu_id in self.p_core_id]
        self.e_core_mask_list = [1 << cpu_id for cpu_id in self.e_core_id]

    def par_run(self, process_list:list[Process], queue_list:list[Queue], log_file, result_fallback=None) -> list:
        '''result_fallback: a list of fallback functions that extract the result if the queue is empty'''
        original_list = process_list.copy()
        result_list = [None]*len(process_list)
        process_queue = process_list[:len(self.available_cpu_id)]
        process_list = process_list[len(self.available_cpu_id):]
        remaining_cpu_id = self.available_cpu_id[len(process_queue):]

        for process in process_queue:
            process.start()
            print("process",process.pid,"started",file=log_file)
        
        pid_list = [proc.pid for proc in process_queue]
        print("current process:",multiprocessing.current_process().pid)
        pid_affinity = assign_cpu(pid_list, self.available_cpu_id)
        print("processes",pid_list,"assigned to",self.available_cpu_id[:len(process_queue)],
                file=log_file)
        
        while process_queue:
            log_file.flush()
            pid_list = [proc.pid for proc in process_queue]
            process_queue_new:list[Process] = []
            for process in process_queue:
                if process.is_alive():
                    try:
                        result = queue_list[original_list.index(process)].get(timeout=5)
                    except Empty:
                        process_queue_new.append(process)
                    else:
                        result_list[original_list.index(process)] = result
                        remaining_cpu_id.append(pid_affinity[process.pid])
                        print("process",process.pid,"closed",file=log_file)
                        process.join(timeout=30)
                        if process.is_alive():
                            process.terminate()
                        if not process.is_alive():
                            process.close()
                        else:
                            print("ERROR:process",process.pid,"fail to terminate",file=log_file)
                else:
                    remaining_cpu_id.append(pid_affinity[process.pid])
                    print("process",process.pid,"closed",file=log_file)
                    try:
                        result = queue_list[original_list.index(process)].get(timeout=5)
                    except Empty:
                        print("process",process.pid,"timeout",file=log_file)
                        result = result_fallback[original_list.index(process)]() if result_fallback else None
                    result_list[original_list.index(process)] = result
                    process.close()
            if len(remaining_cpu_id) > 0 and len(process_list) > 0:
                new_process = process_list[:len(remaining_cpu_id)]
                process_list = process_list[len(remaining_cpu_id):]
                for process in new_process:
                    process.start()
                    print("process",process.pid,"started",file=log_file)

                pid_list = [proc.pid for proc in new_process]
                pid_affinity.update(assign_cpu(pid_list, remaining_cpu_id))
                print("processes",pid_list,"assigned to",remaining_cpu_id[:len(new_process)],
                      file=log_file)
                remaining_cpu_id = remaining_cpu_id[len(new_process):]
                process_queue_new.extend(new_process)
            process_queue = process_queue_new
        return result_list
