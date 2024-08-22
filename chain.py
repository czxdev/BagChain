'''实现BlockHead、Block以及Chain类描述区块链结构'''
import copy
import matplotlib.pyplot as plt
from typing import List

import graphviz
from enum import Enum

from task import Task
from functions import hashG, hashH
import global_var


class BlockHead(object):

    def __init__(self, prehash=None, blockhash=None, timestamp=None, target=None, 
                 nonce=None, height=None, Miner=None):
        self.prehash = prehash  # 前一个区块的hash
        self.timestamp = timestamp  # 时间戳
        self.target = target  # 难度目标
        self.nonce = nonce  # 随机数
        self.height = height  # 高度
        self.blockhash = blockhash  # 区块哈希
        self.miner = Miner  # 矿工
        self.blockheadextra = {}  # 其他共识协议需要用的，使用字典添加
        # 这里有个问题, blockhash靠blockhead自身是算不出来的
        # 块头里面应该包含content的哈希?(其实无所谓)

    # def printblockhead(self):
    #     print("prehash:", self.prehash)
    #     print("blockhash:", self.blockhash)
    #     print("target:", self.target)
    #     print("nouce:", self.nonce)
    #     print("height:", self.height)
    #     print("Miner:", self.miner)
    #     print("timestamp:", self.timestamp)

    # def readlist(self):
    #     return [self.prehash, self.timestamp, self.target, 
    #               self.nonce, self.height, self.blockhash, self.miner]

    # def readstr(self):
    #     s = self.readlist()
    #     data = ''.join([str(x) for x in s])
    #     return data


class Block(object):
    '''描述区块结构，实现区块哈希的计算'''
    class BlockType(Enum):
        '''定义区块类型'''
        MINIBLOCK = 0
        BLOCK = 1 # represent ensemble block
        KEY_BLOCK = 2
    
    class BlockExtra:
        '''描述blockextra中信息'''
        def __init__(self,task_id=0, task_list=None, task_queue = None, miniblock_hash=None, 
                     miniblock_list=None, metric=None, blocktype=None, model=None, 
                     model_hash=None, validate_list=None, ensemble_block_list = None):
            '''BlockExtra对象初始化'''
            self.blocktype = blocktype or Block.BlockType.KEY_BLOCK # 记录区块类型
            self.task_id = task_id
            # 以下为与Ensemble Block相关数据
            self.miniblock_hash:list[str] = miniblock_hash
            self.miniblock_list:BlockList = miniblock_list or []
            self.metric = metric
            # 以下为与Miniblock有关的数据
            self.model_hash = model_hash # 用id(model)代替哈希
            self.model = model
            # 以下为与Key Block相关数据
            self.validate_list:dict = validate_list or {}
            self.ensemble_block_list:BlockList = ensemble_block_list or []
            self.task_list:list[Task] = task_list or []
            self.task_queue:list[Task] = task_queue or []

        def __deepcopy__(self, memo):
            cls = self.__class__
            result = cls.__new__(cls)
            memo[id(self)] = result
            for k,v in self.__dict__.items():
                if cls.__name__ != "BlockExtra" or cls.__name__ == "BlockExtra" and \
                    k != "task_list" and k != "miniblock_list" and k != "model" and \
                    k != "miniblock_hash" and k != "task_queue" and \
                    k != "validate_list" and k != "ensemble_block_list":
                    # 需要进行深复制的数据
                    setattr(result, k, copy.deepcopy(v, memo))
                elif (k == "task_list" or k == "miniblock_list" or \
                     k == "miniblock_hash" or k == "task_queue" or \
                     k == "validate_list" or k == "ensemble_block_list") \
                     and cls.__name__ == "BlockExtra":
                    # 对列表进行浅复制
                    setattr(result, k, copy.copy(v))
                else: # 对model引用不复制
                    setattr(result, k, v)
            return result


    def __init__(self, name=None, blockhead: BlockHead = None, content=None, isadversary=False, 
                 blockextra=None, isgenesis=False, blocksize_MB=2):
        self.name = name
        self.blockhead = blockhead
        self.isAdversaryBlock = isadversary
        self.content = content
        self.next:BlockList = []  # 子块列表
        self.last:Block = None  # 母块
        self.blockextra = blockextra or Block.BlockExtra() # 其他共识协议需要的额外信息
        self.isGenesis = isgenesis
        self.blocksize_MB = blocksize_MB

        # self.blocksize_byte = int(random.uniform(0.5, 2) * 1048576)  # 单位:byte 随机 0.5~1 MB

    def calculate_blockhash(self):
        '''
        计算区块的hash
        return:
            hash type:str
        '''
        prehash = self.blockhead.prehash
        timestamp = self.blockhead.timestamp
        minerid = self.blockhead.miner
        if self.blockextra.blocktype is self.BlockType.MINIBLOCK:
            model_hash = self.blockextra.model_hash
            hash_content = [minerid, timestamp, model_hash, prehash]
        elif self.blockextra.blocktype is self.BlockType.BLOCK: # Ensemble Block
            miniblock_hash_list = self.blockextra.miniblock_hash
            task_id = self.blockextra.task_id
            metric = self.blockextra.metric
            hash_content = [minerid, timestamp, task_id, metric]
            hash_content.append(hashG(miniblock_hash_list))
            hash_content.append(prehash)
        else: # Key Block
            content = self.content
            ensemble_block_hash_list = list(self.blockextra.validate_list.keys())
            metric_list = list(self.blockextra.validate_list.values())
            task_id = self.blockextra.task_id
            metric = self.blockextra.metric
            nonce = self.blockhead.nonce
            hash_content = [minerid, nonce, timestamp, task_id, metric]
            hash_content.append(hashG(ensemble_block_hash_list+metric_list))
            hash_content.append(hashG([prehash, content]))
        return hashH(hash_content)  # 计算哈希


    def printblock(self):
        print('in_rom:', id(self))
        print("blockname:", self.name)
        self.blockhead.printblockhead()
        print("content:", self.content)
        print('isAdversaryBlock:', self.isAdversaryBlock)
        print('next:', self.next)
        print('last:', self.last)
        print('block_extra_id',id(self.blockextra), '\n')

    def ReadBlockHead_list(self):
        return self.blockhead.readlist()

    def ReadBlocHead_str(self):
        return self.blockhead.readstr()

    def BlockHeight(self):
        return self.blockhead.height

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if cls.__name__ == 'Block' and k != 'next' and k != 'last':
                setattr(result, k, copy.deepcopy(v, memo))
            if cls.__name__ == 'Block' and k == 'next':
                setattr(result, k, [])
            if cls.__name__ == 'Block' and k == 'last':
                setattr(result, k, None)
            if cls.__name__ != 'Block':
                setattr(result, k, copy.deepcopy(v, memo))
        return result

BlockList = list[Block]

class Chain(object):

    def __init__(self):

        self.head = None
        self.lastblock = self.head  # 指向最新区块，代表矿工认定的主链

    def __contains__(self, block: Block):
        if not self.head:
            return False
        q = [self.head]
        while q:
            blocktmp = q.pop(0)
            if block.blockhead.blockhash == blocktmp.blockhead.blockhash:
                return True
            for i in blocktmp.next:
                q.append(i)
        return False

    def __iter__(self):
        if not self.head:
            return
        q = [self.head]
        while q:
            blocktmp = q.pop(0)
            yield blocktmp
            for i in blocktmp.next:
                q.append(i)

    def __deepcopy__(self, memo):
        if not self.head:
            return None
        copy_chain = Chain()
        copy_chain.head = copy.deepcopy(self.head)
        memo[id(copy_chain.head)] = copy_chain.head
        q = [copy_chain.head]
        q_o = [self.head]
        copy_chain.lastblock = copy_chain.head
        while q_o:
            for block in q_o[0].next:
                copy_block = copy.deepcopy(block, memo)
                copy_block.last = q[0]
                q[0].next.append(copy_block)
                q.append(copy_block)
                q_o.append(block)
                memo[id(copy_block)] = copy_block
                if block.name == self.lastblock.name:
                    copy_chain.lastblock = copy_block
            q.pop(0)
            q_o.pop(0)
        return copy_chain
    
    def iter_longest_chain(self):
        '''从最新的区块开始迭代'''
        if not self.lastblock:
            return
        block = self.lastblock
        while block:
            yield block
            block = block.last

    def create_genesis_block(self, **blockextra_dict):
        prehash = 0
        time = 0
        target = 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'
        nonce = 0
        height = 0
        Miner_ID = -1  # 创世区块不由任何一个矿工创建
        input = 0
        # currenthash = hashH([Miner_ID, nonce, hashG([prehash, input])])
        task = global_var.get_global_task()
        blockextra = Block.BlockExtra(None, [task], [task,task])
        self.head = Block('K0', BlockHead(prehash, None, time, target, nonce, height, Miner_ID),
                          input, False,blockextra, True)
        self.head.blockhead.blockhash = self.head.calculate_blockhash()
        self.head.blockhead.blockheadextra["value"] = 1  # 不加这一条 其他共识无法运行
        self.lastblock = self.head

    def search(self, block: Block, searchdepth=500):
        # 利用区块哈希，搜索某块是否存在(搜索树)
        # 存在返回区块地址，不存在返回None
        if not self.head or not block:
            return None
        searchroot = self.lastblock
        if block.blockhead.height < searchroot.blockhead.height - searchdepth:
            return None  # 如果搜索的块高度太低 直接不搜了
        i = 0
        while searchroot and searchroot.last and i <= searchdepth:
            if block.blockhead.blockhash == searchroot.blockhead.blockhash:
                return searchroot
            else:
                searchroot = searchroot.last
                i = i + 1
        q = [searchroot]
        while q:
            blocktmp = q.pop(0)
            if block.blockhead.blockhash == blocktmp.blockhead.blockhash:
                return blocktmp
            for i in blocktmp.next:
                q.append(i)
        return None

    def search_chain(self, block: Block, searchdepth=500):
        # 利用区块哈希，搜索某块是否在链上
        # 存在返回区块地址，不存在返回None
        if not self.head:
            return None
        blocktmp = self.lastblock
        i = 0
        while blocktmp and i <= searchdepth:
            if block.blockhead.blockhash == blocktmp.blockhead.blockhash:
                return blocktmp
            blocktmp = blocktmp.last
            i = i + 1
        return None

    def last_block(self):  # 返回最深的block，空链返回None
        return self.lastblock

    def is_empty(self):
        if not self.head:
            print("Chain Is empty")
            return True
        else:
            return False

    def Popblock(self):
        popb = self.last_block()
        last = popb.last
        if not last:
            return None
        else:
            last.next.remove(popb)
            popb.last = None
            return popb

    def add_block_direct(self, block: Block, lastBlock: Block = None, nextBlock: Block = None):
        # 根据定位添加block，如果没有指定位置，加在最深的block后
        # 和AddChain的区别是这个是不拷贝直接连接的
        if self.search(block):
            print("Block {} is already included.".format(block.name))
            return block

        if not self.head:
            self.head = block
            self.lastblock = block
            print("Add Block {} Successfully.".format(block.name))
            return block

        if not lastBlock and not nextBlock:
            last_Block = self.last_block()
            last_Block.next.append(block)
            block.last = last_Block
            self.lastblock = block
            # print("Add Block {} Successfully.".format(block.name))
            return block


    def insert_block_copy(self, copylist: List[Block], insert_point: Block):
        '''在指定的插入点将指定的链合入区块树
        param:
            copylist 待插入的链 type:List[Block]
            insert_point 区块树中的节点，copylist中的链从这里插入 type:Block
        return:
            local_tmp 返回值：深拷贝插入完之后新插入链的块头 type:Block
        '''
        local_tmp = insert_point
        if local_tmp:
            while copylist:
                receive_tmp = copylist.pop()
                blocktmp = copy.deepcopy(receive_tmp)
                blocktmp.last = local_tmp
                blocktmp.next = []
                local_tmp.next.append(blocktmp)
                local_tmp = blocktmp
        return local_tmp  # 返回深拷贝的最后一个区块的指针，如果没拷贝返回None

    def add_block_copy(self, lastblock: Block):
        # 返回值：深拷贝插入完之后新插入链的块头
        receive_tmp = lastblock
        if not receive_tmp:  # 接受的链为空，直接返回
            return None
        copylist = []  # 需要拷贝过去的区块list
        local_tmp = self.search(receive_tmp)
        while receive_tmp and not local_tmp:
            copylist.append(receive_tmp)
            receive_tmp = receive_tmp.last
            local_tmp = self.search(receive_tmp)
        if local_tmp:
            while copylist:
                receive_tmp = copylist.pop()
                blocktmp = copy.deepcopy(receive_tmp)
                blocktmp.last = local_tmp
                blocktmp.next = []
                local_tmp.next.append(blocktmp)
                local_tmp = blocktmp
            if local_tmp.BlockHeight() > self.lastblock.BlockHeight():
                self.lastblock = local_tmp  # 更新global chain的lastblock
        return local_tmp  # 返回深拷贝的最后一个区块的指针，如果没拷贝返回None

    '''
    def Depth(self,method="level"): # 计算chain的最大长度
        if self.head:
            if method == "level":
                q = [(self.head, 1)]
                while q:
                    block, depth = q.pop(0)
                    for i in block.next:
                        q.append((i, depth+1))
            else:
                def ddepth(block:Block):
                    if block.next == []:
                        return 0
                    depth = [0]
                    for i in block.next:
                        depth.append(ddepth(i))
                    return max(depth)+1       
                depth = ddepth(self.head)
        else:
            depth = 0
        return depth
    '''

    def ShowBlock(self):  # 按从上到下从左到右展示block,打印块名
        if not self.head:
            print()
        q = [self.head]
        blocklist = []
        while q:
            block = q.pop(0)
            blocklist.append(block)
            print("{};".format(block.name), end="")
            for i in block.next:
                q.append(i)
        print("")
        return blocklist

    def InversShowBlock(self, block: Block = None):
        # 按指定块为起始逆向打印块名，若未指定块从最深的块开始，返回逆序的链
        cur = self.last_block()
        blocklist = []
        while cur:
            # print(cur.name)
            blocklist.append(cur)
            cur = cur.last
        return blocklist

    def ShowLChain(self):
        # 打印主链
        blocklist = self.InversShowBlock()
        blocklist.reverse()
        for i in blocklist:
            print("{}→→→→".format(i.name), end="")
        print("")
        return blocklist

    '''
    def ShowStructure(self):
        # 打印树状结构
        blocktmp = self.head
        fork_list = []
        while blocktmp:
            print("{}→→→→".format(blocktmp.name), end="")
            list_tmp = copy.copy(blocktmp.next)
            if list_tmp:
                blocktmp = list_tmp.pop(0)
                fork_list.extend(list_tmp)
            else:
                if fork_list:
                    print("")
                    blocktmp = fork_list.pop(0)
                    print("{}→→→→".format(blocktmp.last.name), end="")
                else:
                    blocktmp = None
        print("")
    '''

    def ShowStructure1(self):
        # 打印树状结构
        blocklist = [self.head]
        printnum = 1
        while blocklist:
            length = 0
            print("|    ", end="")
            print("-|   " * (printnum - 1))
            while printnum > 0:
                blocklist.extend(blocklist[0].next)
                blockprint = blocklist.pop(0)
                length += len(blockprint.next)
                print("{}   ".format(blockprint.name), end="")
                printnum -= 1
            print("")
            printnum = length

    def ShowStructure(self, miner_num=10):
        # 打印树状结构
        # 可能需要miner数量 也许放在这里不是非常合适？
        plt.figure()
        blocktmp = self.head
        fork_list = []
        while blocktmp:
            if blocktmp.isGenesis is False:
                rd2 = blocktmp.content + blocktmp.blockhead.miner / miner_num
                rd1 = blocktmp.last.content + blocktmp.last.blockhead.miner / miner_num
                ht2 = blocktmp.blockhead.height
                ht1 = ht2 - 1
                if blocktmp.isAdversaryBlock:
                    plt.scatter(rd2, ht2, color='r', marker='o')
                    plt.plot([rd1, rd2], [ht1, ht2], color='r')
                else:
                    plt.scatter(rd2, ht2, color='b', marker='o')
                    plt.plot([rd1, rd2], [ht1, ht2], color='b')
            else:
                plt.scatter(0, 0, color='b', marker='o')
            list_tmp = copy.copy(blocktmp.next)
            if list_tmp:
                blocktmp = list_tmp.pop(0)
                fork_list.extend(list_tmp)
            else:
                if fork_list:
                    blocktmp = fork_list.pop(0)
                else:
                    blocktmp = None
        plt.xlabel('round')
        plt.ylabel('block height')
        plt.title('blockchain visualisation')
        plt.grid(True)
        RESULT_PATH = global_var.get_result_path()
        plt.savefig(RESULT_PATH / 'blockchain visualisation.svg')
        if global_var.get_show_fig():
            plt.show()
        plt.close()

    def ShowStructureWithGraphviz(self):
        '''借助Graphviz将区块链可视化'''
        # 采用有向图
        dot = graphviz.Digraph('Blockchain Structure',engine='dot')
        blocktmp = self.head
        fork_list = []
        miniblock_name_list = []
        ensemble_block_name_list = []
        while blocktmp:
            if blocktmp.isGenesis is False:
                # 建立Key Block节点
                if blocktmp.isAdversaryBlock:
                    dot.node(blocktmp.name, shape='rect', color='red',
                             label=blocktmp.name+':'+str(round(blocktmp.blockextra.metric, 4)))
                else:
                    dot.node(blocktmp.name,shape='rect',color='orange',
                             label=blocktmp.name+':'+str(round(blocktmp.blockextra.metric, 4)))
                # 建立Ensemble Block节点
                for ensemble_block in blocktmp.blockextra.ensemble_block_list:
                    if ensemble_block.name not in ensemble_block_name_list:
                        dot.node(ensemble_block.name, shape='rect', color='yellow',
                                 label=ensemble_block.name+':'+str(round(ensemble_block.blockextra.metric, 4)))
                        ensemble_block_name_list.append(ensemble_block.name)
                        # 建立Ensemble Block与Miniblock的连接（去重）
                        for miniblock in ensemble_block.blockextra.miniblock_list:
                            dot.edge(miniblock.name, ensemble_block.name)
                    # 建立Ensemble Block与Key Block的连接
                    if blocktmp.blockextra.validate_list[ensemble_block.blockhead.blockhash] \
                        == blocktmp.blockextra.metric:
                        dot.edge(ensemble_block.name, blocktmp.name, color='orange')
                    else:
                        dot.edge(ensemble_block.name, blocktmp.name)
                    # 建立Miniblock节点
                    for miniblock in ensemble_block.blockextra.miniblock_list:
                        if miniblock.name not in miniblock_name_list:
                            dot.node(miniblock.name, shape='rect', color='green')
                            # 建立上一高度Key Block与Miniblock的连接（去重）
                            dot.edge(miniblock.last.name, miniblock.name)
                            miniblock_name_list.append(miniblock.name)

            else:
                dot.node('K0',shape='rect',color='black',fontsize='20')
            list_tmp = copy.copy(blocktmp.next)
            if list_tmp:
                blocktmp = list_tmp.pop(0)
                fork_list.extend(list_tmp)
            else:
                if fork_list:
                    blocktmp = fork_list.pop(0)
                else:
                    blocktmp = None
        # 生成矢量图,展示结果
        dot.render(directory=global_var.get_result_path() / "blockchain_visualization",
                   format='svg', view=global_var.get_show_fig())
    
    def ShowStructureWithGraphvizWithEverything(self, complete_miniblock_list:BlockList,
                                                complete_ensemble_block_list:BlockList):
        '''借助Graphviz将区块链可视化'''
        # 采用有向图
        dot = graphviz.Digraph('Blockchain Structure',engine='dot')
        blocktmp = self.head
        fork_list = []
        miniblock_name_list = []
        ensemble_block_name_list = []
        while blocktmp:
            if blocktmp.isGenesis == False:
                # 建立Key Block节点
                if blocktmp.isAdversaryBlock:
                    dot.node(blocktmp.name, shape='rect', color='red',
                             label=blocktmp.name+':'+str(round(blocktmp.blockextra.metric, 4)))
                else:
                    dot.node(blocktmp.name,shape='rect',color='orange',
                             label=blocktmp.name+':'+str(round(blocktmp.blockextra.metric, 4)))
                # 建立Ensemble Block节点
                for ensemble_block in blocktmp.blockextra.ensemble_block_list:
                    if ensemble_block.name not in ensemble_block_name_list:
                        dot.node(ensemble_block.name, shape='rect', color='yellow',
                                 label=ensemble_block.name+':'+str(round(ensemble_block.blockextra.metric, 4)))
                        ensemble_block_name_list.append(ensemble_block.name)
                        # 建立Ensemble Block与Miniblock的连接（去重）
                        for miniblock in ensemble_block.blockextra.miniblock_list:
                            dot.edge(miniblock.name, ensemble_block.name)
                    # 建立Ensemble Block与Key Block的连接
                    if blocktmp.blockextra.validate_list[ensemble_block.blockhead.blockhash] \
                        == blocktmp.blockextra.metric:
                        dot.edge(ensemble_block.name, blocktmp.name, color='orange')
                    else:
                        dot.edge(ensemble_block.name, blocktmp.name)
                    # 建立Miniblock节点
                    for miniblock in ensemble_block.blockextra.miniblock_list:
                        if miniblock.name not in miniblock_name_list:
                            dot.node(miniblock.name, shape='rect', color='green')
                            # 建立上一高度Key Block与Miniblock的连接（去重）
                            dot.edge(miniblock.last.name, miniblock.name)
                            miniblock_name_list.append(miniblock.name)
            else:
                dot.node('K0',shape='rect',color='black',fontsize='20')
            list_tmp = copy.copy(blocktmp.next)
            if list_tmp:
                blocktmp = list_tmp.pop(0)
                fork_list.extend(list_tmp)
            else:
                if fork_list:
                    blocktmp = fork_list.pop(0)
                else:
                    blocktmp = None
        # 查漏补缺
        for miniblock in complete_miniblock_list:
            if miniblock.name not in miniblock_name_list:
                dot.node(miniblock.name, shape='rect', color='green')
                dot.edge(miniblock.last.name, miniblock.name)
        for ensemble_block in complete_ensemble_block_list:
            if ensemble_block.name not in ensemble_block_name_list:
                dot.node(ensemble_block.name, shape='rect', color='yellow',
                         label=ensemble_block.name+':'+str(ensemble_block.blockextra.metric))
                for miniblock in ensemble_block.blockextra.miniblock_list:
                    dot.edge(miniblock.name, ensemble_block.name)
        # 生成矢量图,展示结果
        dot.render(filename="blockchain_visualization_with_stale_intermediate_blocks",
                   directory=global_var.get_result_path() / "blockchain_visualization",
                   format='svg', view=global_var.get_show_fig())

    def get_block_interval_distribution(self):
        if self.lastblock.blockhead.height == 0:
            return
        stat = []
        blocktmp2 = self.lastblock
        while not blocktmp2.isGenesis:
            blocktmp1 = blocktmp2.last
            stat.append(blocktmp2.content - blocktmp1.content)
            blocktmp2 = blocktmp1
        plt.hist(stat, bins=10, histtype='bar', range=(0, max(stat)))
        plt.xlabel('Rounds')
        plt.ylabel('Times')
        plt.title('Block generation interval distribution')
        RESULT_PATH = global_var.get_result_path()
        plt.savefig(RESULT_PATH / 'block interval distribution.svg')
        if global_var.get_show_fig():
            plt.show()
        plt.close()


    def CalculateStatistics(self, rounds):
        # 统计一些数据
        stats = {
            "num_of_generated_blocks": -1,
            "num_of_valid_blocks": 0,
            "num_of_stale_blocks": 0,
            "stale_rate": 0,
            "num_of_forks": 0,
            "fork_rate": 0,
            "average_block_time_main": 0,
            "block_throughput_main": 0,
            "throughput_main_MB": 0,
            "average_block_time_total": 0,
            "block_throughput_total": 0,
            "throughput_total_MB": 0
        }
        q = [self.head]
        while q:
            stats["num_of_generated_blocks"] = stats["num_of_generated_blocks"] + 1
            blocktmp = q.pop(0)
            if blocktmp.blockhead.height > stats["num_of_valid_blocks"]:
                stats["num_of_valid_blocks"] = blocktmp.blockhead.height
            nextlist = blocktmp.next
            q.extend(nextlist)

        last_block = self.lastblock.last
        while last_block:
            stats["num_of_forks"] += len(last_block.next) - 1
            last_block = last_block.last

        stats["num_of_stale_blocks"] = stats["num_of_generated_blocks"] - stats["num_of_valid_blocks"]
        stats["average_block_time_main"] = rounds / stats["num_of_valid_blocks"]
        stats["block_throughput_main"] = 1 / stats["average_block_time_main"]
        blocksize = global_var.get_blocksize()
        stats["throughput_main_MB"] = blocksize * stats["block_throughput_main"]
        stats["average_block_time_total"] = rounds / stats["num_of_generated_blocks"]
        stats["block_throughput_total"] = 1 / stats["average_block_time_total"]
        stats["throughput_total_MB"] = blocksize * stats["block_throughput_total"]
        stats["fork_rate"] = stats["num_of_forks"] / stats["num_of_generated_blocks"]
        stats["stale_rate"] = stats["num_of_stale_blocks"] / stats["num_of_generated_blocks"]

        return stats


if __name__ == '__main__':
    blocknew = Block(1, BlockHead(111, 111, 111, 111, 111, 111, 1), 1, False)
    blocknew.next = [1, 1, 1]
    blocknew.last = 1
    block2 = copy.deepcopy(blocknew)
    blocknew.printblock()
    block2.printblock()

    print(blocknew.calculate_blockhash())
    blocknew.blockextra.blocktype = Block.BlockType.MINIBLOCK
    print(blocknew.calculate_blockhash())

    print(Block.BlockExtra.__name__)
    print("==============")
    task_list_test = [1,2,3,[123,431,2]]
    model_test = [2,3,3,1]
    blockextra_test = Block.BlockExtra(id(task_list_test), task_list_test, model=model_test)
    blockextra_deepcopy = copy.deepcopy(blockextra_test)
    print(id(blockextra_test.model))
    print(id(blockextra_deepcopy.model))
    print(id(task_list_test))
    print(id(blockextra_test.task_list))
    print(id(blockextra_deepcopy.task_list))
    print(id(blockextra_test.task_list[-1]))
    print(id(blockextra_deepcopy.task_list[-1]))
