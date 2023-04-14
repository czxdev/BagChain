from abc import ABCMeta, abstractmethod

from chain import Block
from functions import hashG, hashH

class Consensus(metaclass=ABCMeta):        #抽象类

    @abstractmethod
    def setparam(self):
        '''设置共识所需参数'''
        pass

    @abstractmethod
    def mining_consensus(self):
        '''共识机制定义的挖矿算法
        return:
            新产生的区块  type:Block 
            挖矿成功标识    type:bool
        '''
        pass

    @abstractmethod
    def valid_chain(self):
        '''检验链是否合法
        return:
            合法标识    type:bool
        '''
        pass

    @abstractmethod
    def valid_block(self):
        '''检验单个区块是否合法
        return:合法标识    type:bool
        '''
        pass