import hashlib
import importlib


def hashsha256(contentlist:list)->str:
    '''
    计算哈希值
    输入:需要转化为hash的内容           type:list
    输出:该list中所有内容的十六进制hash  type:str
    '''
    s = hashlib.sha256()
    data = ''.join([str(x) for x in contentlist])
    s.update(data.encode("utf-8")) 
    b = s.hexdigest()
    return b

def hashH(contentlist:list)->str:
    # 文章中hash function定义了两个，虽然对于bitcoin是一样的，但还是假模假样搞一下
    return hashsha256(contentlist)

def hashG(contentlist:list)->str:
    return hashsha256(contentlist)

def for_name(name):
    """
    返回并加载指定的类
    输入:name为指定类的路径 如Consensus.POW    type: str
    """
    class_name = name.split('.')[-1]
    file = importlib.import_module(name[:name.index("." + class_name)])
    clazz = getattr(file, class_name)
 
    return clazz
    
def primestream(n):
    # 质数迭代生成器，n为给定范围
    lst = [True]*(n+1)
    for i in range(2,n+1):
        if lst[i]:
            yield i
            for j in range(i, n//i+1):
                lst[i*j] = False


def targetG(p_per_round,miner_num,group,q):
    '''
    p = target/group
    (1-p)^(miner_num*q)=1 - p_per_round
    1-p=(1-p_per_round)**(1/(miner_num*q))
    p=1-(1-p_per_round)**(1/(miner_num*q))
    target = round(group*p)
    '''
    p=1-(1-p_per_round)**(1/(miner_num*q))
    target = round(group*p)
    return hex(target)[2:]


def target_adjust(difficulty):
    '''
    根据难度调整target前导零的数量,难度越大前导零越少
    param:difficulty(float):0~1
    return:target(str):256位的16进制数
    '''
    difficulty = difficulty
    leading_zeros_num = int(difficulty * (64 - 1))  # 前导零数量
    target = '0' * leading_zeros_num + 'F' * (64 - leading_zeros_num)  # 在前导零之前插入0字符
    return target

 
# global_var._init()
# #global_var.set_consensus_type("dbprt")
# print(global_var.get_consensus_type())
# # Test = for_name("testforname.Test")
# # print(Test)
# tt = for_name("testforname.Test")("dede",  18)
# tt =testforname.Test("dede",  18)
# print(tt.get_name())
# print(tt.get_age())
