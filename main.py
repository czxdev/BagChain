from Environment import Environment
import time
import logging
import global_var
import configparser


def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

# 读取配置文件
config = configparser.ConfigParser()
config.optionxform = lambda option: option
config.read('system_config.ini',encoding='utf-8')
environ_settings = dict(config['EnvironmentSettings'])

# 设置全局变量
global_var.__init__()
global_var.set_consensus_type(environ_settings['consensus_type'])
global_var.set_network_type(environ_settings['network_type'])
global_var.set_miner_num(int(environ_settings['miner_num']))
global_var.set_ave_q(int(environ_settings['q_ave']))
global_var.set_blocksize(int(environ_settings['blocksize']))
global_var.set_show_fig(False)
global_var.save_configuration()

# 配置日志文件
logging.basicConfig(filename=global_var.get_result_path() / 'events.log',
                    level=global_var.get_log_level(), filemode='w')

# 设置网络参数
network_param = {}
if environ_settings['network_type'] == 'network.TopologyNetwork':
    network_param = {'TTL':config.getint('TopologyNetworkSettings', 'TTL'),
                     'gen_net_approach': config.get('TopologyNetworkSettings', 'gen_net_approach'),
                     'save_routing_graph': config.getboolean('TopologyNetworkSettings', 'save_routing_graph'),
                     'ave_degree': config.getfloat('TopologyNetworkSettings', 'ave_degree'),
                     'show_label': config.getboolean('TopologyNetworkSettings', 'show_label')}
elif environ_settings['network_type'] == 'network.BoundedDelayNetwork':
    network_param = {k:float(v) for k,v in dict(config['BoundedDelayNetworkSettings']).items()}

# 生成环境
t = int(environ_settings['t'])
q_ave = int(environ_settings['q_ave'])
q_distr = environ_settings['q_distr']
target = environ_settings['target']
adversary_ids = eval(environ_settings['adversary_ids'])
Z = Environment(t, q_ave, q_distr, target, network_param, *adversary_ids)


@get_time
def run():
    Z.exec(int(environ_settings['total_round']))

    Z.view()

run()
