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

config = configparser.ConfigParser()
config.optionxform = lambda option: option
config.read('system_config.ini',encoding='utf-8')
print(config['EnvironmentSettings']['q'])
environ_settings = dict(config['EnvironmentSettings'])

global_var.__init__()
global_var.set_consensus_type(environ_settings['consensus_type'])
global_var.set_network_type(environ_settings['network_type'])
global_var.set_miner_num(int(environ_settings['miner_num']))
global_var.set_qmax(int(environ_settings['q']))
global_var.set_blocksize(int(environ_settings['blocksize']))
global_var.set_show_fig(False)
global_var.save_configuration()

# 配置日志文件
logging.basicConfig(filename=global_var.get_result_path() / 'events.log',
                    level=global_var.get_log_level(), filemode='w')

adversary_ids = eval(environ_settings['adversary_ids'])
network_param = {}
if environ_settings['network_type'] == 'network.TopologyNetwork':
    network_param = {'TTL':config.getint('TopologyNetworkSettings','TTL'),
                     'gen_net_approach':config.get('TopologyNetworkSettings','gen_net_approach'),
                     'save_routing_graph':config.getboolean('TopologyNetworkSettings','save_routing_graph')}
elif environ_settings['network_type'] == 'network.BoundedDelayNetwork':
    network_param = {k:float(v) for k,v in dict(config['BoundedDelayNetworkSettings'])}

Z = Environment(int(environ_settings['t']), int(environ_settings['q']),environ_settings['target'], network_param, *adversary_ids)


@get_time
def run():
    Z.exec(10000)

    Z.view()

run()
