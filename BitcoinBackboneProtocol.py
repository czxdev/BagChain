# Bitcoin Backbone Protocol
from external import I

def BitcoinBackboneProtocol(honest_miner, round):
    chain_update, update_index = honest_miner.maxvalid()
    # if input contains READ:
    # write R(Cnew) to OUTPUT() 这个output不知道干什么用的
    honest_miner.input = I(round, honest_miner.input_tape)  # I function
    #print("outer2",honest_miner.input)
    newblock, mine_success = honest_miner.Mining()
    #print("outer3",honest_miner.input)
    if update_index or mine_success:  # Cnew != C
        return newblock
    else:
        return None  #  如果没有更新 返回空告诉environment回合结束