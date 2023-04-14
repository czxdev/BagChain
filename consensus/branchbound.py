import sys
import math
from typing import List

import numpy as np
from scipy.optimize import linprog

import global_var
from chain import Block, BlockHead, Chain

from .consensus_abc import Consensus


class LpPrblm(object):
    def __init__(self, c, G_ub, h_ub, A_eq, b_eq, bounds, x_nk):
        '''
        maximize:c @ x
        subject to:
            G_ub @ x <= h_ub
            A_eq @ x == b_eq
            lb <= x <= ub
        '''
        # coefficients
        self.c = c
        self.G_ub = G_ub
        self.h_ub = h_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.bounds = bounds
        # index of the variable branched out
        self.x_nk = x_nk
        # results
        self.success = None #feasible or not
        self.x_lp = None # linear optimal solution
        self.z_lp = None # objective value(upper bound of P_h)

    # Slove the LP
    def solve_lp(self):
        r = linprog(self.c, self.G_ub, self.h_ub, self.A_eq, self.b_eq, self.bounds)
        self.success = r.success
        if r.success:
            self.x_lp = r.x
            self.z_lp = -r.fun
        return self.x_lp, self.z_lp, r.success


class BranchBound(Consensus):
    def __init__(self):
        self.key_block_orig = Block()
        self.cur_block = Block()
        # self.cur_prblm_1 = {}
        self.fathomed_blocks:List[Block] = []
        self.upper_bound = sys.maxsize
        self.lower_bound = -sys.maxsize

    def mining_consensus(self, blockchain, miner_id, isAdversary, input, q):
        ...
    
    def valid_chain(self, lastblock: Block):
        ...

    def valid_block(self, block: Block):
        ...
    
    def gen_keyblock_orig(self, blockchain:Chain, miner_id, c, A_eq, b_eq, G_ub, h_ub, bounds):
        '''Receive a ILP from environment and genearte origin keyblock.'''
        # gen orig LpPrblm and automatically solve it
        orig_prblm = LpPrblm(c, G_ub, h_ub, A_eq, b_eq, bounds)
        success_flag = orig_prblm.solve_lp()
        if success_flag:
            # gen keyblock B_{h}
            height = blockchain.lastblock.blockhead.height
            block_name = f'B{str(global_var.get_block_number())}'
            self.key_block = Block(block_name,
                BlockHead(blockchain.lastblock.name, block_name, None, None ,None, height+1,miner_id),
                None, False, False, 0)
            self.key_block.blockextra = {'is_keyblock': True, 'orig_prblm': orig_prblm}
        else:
            self.key_block = None
        return success_flag, self.key_block

    def gen_miniblock(self, miner_id, cur_block:Block, c, A_eq, b_eq, new_G_ub1, new_G_ub2, new_h_ub1, new_h_ub2, h_ub, bounds):
        '''If solve both the subproblems, gen a miniblock.'''
        subprblm_1 =  LpPrblm(c, new_G_ub1, new_h_ub1, A_eq, b_eq, bounds)
        subprblm_2 =  LpPrblm(c, new_G_ub2, new_h_ub2, A_eq, b_eq, bounds)
        height = cur_block.blockhead.height
        block_name = f'B{str(global_var.get_block_number())}'
        mini_block =  Block(block_name,
                    BlockHead(cur_block.name, block_name, None, None ,None, height+1, miner_id),
                    None, False, False, global_var.get_blocksize())
        mini_block.blockextra = {
            'is_keyblock': True,
            'subprblm_1': subprblm_1,
            'subprblm_2': subprblm_2
        }
        return mini_block

    def gen_keyblock_solved():
        ...

    def select_subprblm_var(self, cur_block:Block):
        '''Select a subprblm from keyblock or miniblock and select a var to branch.'''
        # If miniblock, select a subprblm randomly
        if cur_block.blockextra['is_keyblock'] is True:
            cur_prblm:LpPrblm = self.cur_block.blockextra['orig_prblm']
        else:
            if np.random.rand() < 0.5:
                cur_prblm:LpPrblm = self.cur_block.blockextra['subprblm_1']
            else:
                cur_prblm:LpPrblm = self.cur_block.blockextra['subprblm_2']

        # Selct a varible to branch -- the frist non-integer
        for idx, x in enumerate(cur_prblm.x_lp):
            if not x.is_integer():
                break
        return cur_prblm, idx

    def branch(self, cur_prblm:LpPrblm, idx):
        '''Generate two subproblems and solve them.'''
        # new constraints
        new_con1 = np.zeros(cur_prblm.G_ub.shape[1])
        new_con1[idx] = -1
        new_con2 = np.zeros(cur_prblm.G_ub.shape[1])
        new_con2[idx] = 1
        new_G_ub_loc = cur_prblm.G_ub.shape[0]
        new_h_ub_loc = cur_prblm.h_ub.shape[0]
        new_G_ub1 = np.insert(cur_prblm.G_ub, new_G_ub_loc, new_con1, axis=0)
        new_G_ub2 = np.insert(cur_prblm.G_ub, new_G_ub_loc, new_con2, axis=0)
        new_h_ub1 = np.insert(cur_prblm.h_ub, new_h_ub_loc, -math.ceil(cur_prblm.x[idx]), axis=0)
        new_h_ub2 = np.insert(cur_prblm.h_ub, new_h_ub_loc, math.floor(cur_prblm.x[idx]), axis=0)
        # gen new subproblems
        new_prblm1 = LpPrblm(cur_prblm.c, new_G_ub1, new_h_ub1, cur_prblm.A_eq, cur_prblm.b_eq, cur_prblm.bounds, idx)
        new_prblm2 = LpPrblm(cur_prblm.c, new_G_ub2, new_h_ub2, cur_prblm.A_eq, cur_prblm.b_eq, cur_prblm.bounds, idx)
        r1 = self.solve_lp(new_prblm1)
        r2 = self.solve_lp(new_prblm2)
        return r1, r2

    def solve_lp(self, lp_prblm:LpPrblm):
        '''Solve the given lp problem.'''
        r = linprog(lp_prblm.c, lp_prblm.G_ub, lp_prblm.h_ub, 
                    lp_prblm.A_eq, lp_prblm.b_eq, lp_prblm.bounds)
        lp_prblm.success = r.success
        if r.success:
            lp_prblm.x_lp = r.x
            lp_prblm.z_lp = -r.fun
        return r
            
    def bound():
        ...

    
    def prune_branch_fathomed(self):
        ...


if __name__ == '__main__':
    import os
    curPath = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(curPath)    


