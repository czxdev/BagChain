from scipy.optimize import linprog
import numpy as np
import math
import sys
from queue import Queue

class BBNode(object):
    def __init__(self, fun, x, A_ub, b_ub, node_id):
        self.fun = fun
        self.x = x
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.node_id = node_id
        self.last = None
        self.next = []

    def addlast(self,bbnode):
        self.last = bbnode
    
    def addnext(self,bbnode):
        self.next.append(bbnode)

    def shownode(self):
        print('node_id: {}\n fun: {}\n x: {}\n A_ub: {}\n b_ub: {}'.format(
            self.node_id, self.fun, self.x, self.A_ub, self.b_ub))
        if self.next is not []:
            print('next_nodes:',[n.node_id for n in self.next])
        else: 
            print('next:[]')
        if self.last is not None:
            print('last_node:',self.last.node_id,'\n')
        else:
            print('last_node: None\n')


class BranchandBound():
    def __init__(self, c, A_ub, b_ub, A_eq, b_eq, bounds):
        # 全局参数
        self.LOWER_BOUND = -sys.maxsize
        self.UPPER_BOUND = sys.maxsize
        self.opt_val = None
        self.opt_x = None
        self.opt_node = None
        self.Q = Queue()#广度优先
        self.deepest_node_id = 0

        # 这些参数在每轮计算中都不会改变
        self.c = -c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.bounds = bounds

        # 首先计算一下初始问题
        r = linprog(-c, A_ub, b_ub, A_eq, b_eq, bounds)

        # 若最初问题线性不可解
        if not r.success:
            raise ValueError('Not a feasible problem!')

        # 将解和约束参数放入队列
        self.head = BBNode(r.fun, r.x, A_ub, b_ub, self.deepest_node_id)
        self.Q.put(self.head)



    def addnode(self,last_node,new_node):
        last_node.addnext(new_node)
        new_node.addlast(last_node)


    def solve(self):
        while not self.Q.empty():
            
            print('Queue: ',[f'node{bbnode.node_id}' for bbnode in self.Q.queue])
           
            # 取出当前问题
            cur_node = self.Q.get(block=False)

            # 当前最优值小于总下界，则排除此区域 （剪枝）
            if -cur_node.fun < self.LOWER_BOUND:
                continue

            # 若结果 x 中全为整数，则尝试更新全局下界、全局最优值和最优解（定界）
            if all(list(map(lambda f: f.is_integer(), cur_node.x))):
                if self.LOWER_BOUND < -cur_node.fun:
                    self.LOWER_BOUND = -cur_node.fun

                if self.opt_val is None or self.opt_val < -cur_node.fun:
                    self.opt_val = -cur_node.fun
                    self.opt_x = cur_node.x
                    self.opt_node = cur_node
                    print(f'opt_val: {self.opt_val}, opt_x: {self.opt_x}, opt_node: {cur_node.node_id}')
                continue

            # 进行分枝
            else:
            # 寻找 x 中第一个不是整数的，取其下标 idx
                idx = 0
                for i, x in enumerate(cur_node.x):
                    if not x.is_integer():
                        break
                    idx += 1

                # 构建新的约束条件
                new_con1 = np.zeros(cur_node.A_ub.shape[1])
                new_con1[idx] = -1
                new_con2 = np.zeros(cur_node.A_ub.shape[1])
                new_con2[idx] = 1
                new_A_ub1 = np.insert(cur_node.A_ub, cur_node.A_ub.shape[0], new_con1, axis=0)
                new_A_ub2 = np.insert(cur_node.A_ub, cur_node.A_ub.shape[0], new_con2, axis=0)
                new_b_ub1 = np.insert(cur_node.b_ub, cur_node.b_ub.shape[0], -math.ceil(cur_node.x[idx]), axis=0)
                new_b_ub2 = np.insert(cur_node.b_ub, cur_node.b_ub.shape[0], math.floor(cur_node.x[idx]), axis=0)

                # 将新约束条件加入队列，先加最优值大的那一支               
                r1 = linprog(self.c, new_A_ub1, new_b_ub1, self.A_eq, self.b_eq, self.bounds)
                r2 = linprog(self.c, new_A_ub2, new_b_ub2, self.A_eq, self.b_eq, self.bounds)
                self.deepest_node_id+=1
                newnode1 = BBNode(r1.fun, r1.x, new_A_ub1, new_b_ub1, self.deepest_node_id)
                self.deepest_node_id+=1
                newnode2 = BBNode(r2.fun, r2.x, new_A_ub2, new_b_ub2, self.deepest_node_id)
                self.addnode(cur_node,newnode1)
                self.addnode(cur_node,newnode2)
              
                #不可行的剪枝，可行的留下,加入队列Queue
                if not r1.success and r2.success:
                    self.Q.put(newnode2)
                elif not r2.success and r1.success:
                    self.Q.put(newnode1)
                elif r1.success and r2.success:
                    if -r1.fun > -r2.fun:
                        self.Q.put(newnode1)
                        self.Q.put(newnode2)
                    else:
                        self.Q.put(newnode2)
                        self.Q.put(newnode1)

    def show_solvetree(self):  # 按从上到下从左到右展示block,打印块名
        print('\n>>>>>>>>>>>> show solve tree >>>>>>>>>>>> ')
        if not self.head:
            print('Empty')
            return
        q = [self.head]
        nodelist = []
        while q:
            node = q.pop(0)
            nodelist.append(node)
            node.shownode()
            for i in node.next:
                q.append(i)
        if self.opt_node:
            print(f'opt_val: {self.opt_val}, opt_x: {self.opt_x}, opt_node: {self.opt_node.node_id}\n')
        print('>>>>>>>>>>>> solve tree end >>>>>>>>>>>>\n')




def test1():
    """ 此测试的真实最优解为 [4, 2] """
    c = np.array([40, 90])
    A = np.array([[9, 7], [7, 20]])
    b = np.array([56, 70])
    Aeq = None
    beq = None
    bounds = [(0, None), (0, None)]

    solver = BranchandBound(c, A, b, Aeq, beq, bounds)
    solver.solve()
    solver.show_solvetree()
    print("Test 1's result:", solver.opt_val, solver.opt_x)


def test2():
    """ 此测试的真实最优解为 [2, 4] """
    c = np.array([3, 13])
    A = np.array([[2, 9], [11, -8]])
    b = np.array([40, 82])
    Aeq = None
    beq = None
    bounds = [(0, None), (0, None)]

    solver = BranchandBound(c, A, b, Aeq, beq, bounds)
    solver.solve()
    solver.show_solvetree()
    print("Test 2's result:", solver.opt_val, solver.opt_x)


def test3():
    c = np.array([4, 3])
    A = np.array([[4, 5], [2, 1]])
    b = np.array([20, 6])
    Aeq = None
    beq = None
    bounds = [(0, None), (0, None)]

    solver = BranchandBound(c, A, b, Aeq, beq, bounds)
    solver.solve()
    solver.show_solvetree()
    print("Test 3's result:", solver.opt_val, solver.opt_x, solver.LOWER_BOUND)

if __name__ == '__main__':
    test2()