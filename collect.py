"""自我对弈收集数据"""
import random
from collections import deque
import copy
import os
import pickle
import time
from game import Board, Game, move_action2move_id, move_id2move_action, flip_map
from mcts import MCTSPlayer
from config import CONFIG

if CONFIG['use_redis']:
    import my_redis, redis

import zip_array

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')


# 定义整个对弈收集数据流程
class CollectPipeline:

    def __init__(self, init_model=None):
        # 象棋逻辑和棋盘
        self.board = Board()
        self.game = Game(self.board)
        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = CONFIG['play_out']  # 每次移动的模拟次数
        self.c_puct = CONFIG['c_puct']  # u的权重
        self.buffer_size = CONFIG['buffer_size']  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        if CONFIG['use_redis']:
            self.redis_cli = my_redis.get_redis_cli()

    # 从主体加载模型
    def load_model(self):
        if CONFIG['use_frame'] == 'paddle':
            model_path = CONFIG['paddle_model_path']
        elif CONFIG['use_frame'] == 'pytorch':
            model_path = CONFIG['pytorch_model_path']
        else:
            print('暂不支持所选框架')
        try:
            self.policy_value_net = PolicyValueNet(model_file=model_path)
            print('已加载最新模型')
        except:
            self.policy_value_net = PolicyValueNet()
            print('已加载初始模型')
        # 启用批量推理以加速（从config读取batch_size）
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1,
                                      batch_size=CONFIG.get('mcts_batch_size', 16))

    def get_equi_data(self, play_data):
        """左右对称变换，扩充数据集一倍，加速一倍训练速度"""
        extend_data = []
        # 棋盘状态shape is [9, 10, 9], 走子概率，赢家
        for state, mcts_prob, winner in play_data:
            # 原始数据
            extend_data.append(zip_array.zip_state_mcts_prob((state, mcts_prob, winner)))
            # 水平翻转后的数据
            state_flip = state.transpose([1, 2, 0])
            state = state.transpose([1, 2, 0])
            for i in range(10):
                for j in range(9):
                    state_flip[i][j] = state[i][8 - j]
            state_flip = state_flip.transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                mcts_prob_flip[i] = mcts_prob[move_action2move_id[flip_map(move_id2move_action[i])]]
            extend_data.append(zip_array.zip_state_mcts_prob((state_flip, mcts_prob_flip, winner)))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        # 确保数据目录存在
        data_dir = CONFIG.get('data_dir', 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f'已创建数据目录: {data_dir}')
        
        # 收集自我对弈的数据
        for i in range(n_games):
            self.load_model()  # 从本体处加载最新模型
            # 从配置读取是否显示可视化
            is_shown = CONFIG.get('show_selfplay', False)
            show_delay = CONFIG.get('show_delay', 0.5)
            winner, play_data = self.game.start_self_play(
                self.mcts_player, 
                temp=self.temp, 
                is_shown=is_shown,
                show_delay=show_delay
            )
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 增加数据
            play_data = self.get_equi_data(play_data)
            if CONFIG['use_redis']:
                while True:
                    try:

                        for d in play_data:
                            self.redis_cli.rpush('train_data_buffer', pickle.dumps(d))
                        self.redis_cli.incr('iters')
                        self.iters = self.redis_cli.get('iters')
                        print("存储完成")
                        break
                    except:
                        print("存储失败")
                        time.sleep(1)
            else:
                if os.path.exists(CONFIG['train_data_buffer_path']):
                    while True:
                        try:
                            with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                                data_file = pickle.load(data_dict)
                                self.data_buffer = deque(maxlen=self.buffer_size)
                                self.data_buffer.extend(data_file['data_buffer'])
                                self.iters = data_file['iters']
                                del data_file
                                self.iters += 1
                                self.data_buffer.extend(play_data)
                            print('成功载入数据')
                            break
                        except:
                            time.sleep(30)
                else:
                    self.data_buffer.extend(play_data)
                    self.iters += 1
            data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
            with open(CONFIG['train_data_buffer_path'], 'wb') as data_file:
                pickle.dump(data_dict, data_file)
        return self.iters

    def run(self):
        """开始收集数据"""
        try:
            while True:
                iters = self.collect_selfplay_data()
                print('batch i: {}, episode_len: {}'.format(
                    iters, self.episode_len))
        except KeyboardInterrupt:
            print('\n\n收到停止信号（Ctrl+C），正在保存数据...')
            # 确保数据已保存（collect_selfplay_data 中已经保存，这里再次确认）
            if not CONFIG['use_redis']:
                try:
                    if os.path.exists(CONFIG['train_data_buffer_path']):
                        with open(CONFIG['train_data_buffer_path'], 'rb') as f:
                            data = pickle.load(f)
                            print(f'数据已保存到: {CONFIG["train_data_buffer_path"]}')
                            print(f'  总对局数: {data.get("iters", 0)}')
                            print(f'  数据样本数: {len(data.get("data_buffer", []))}')
                    else:
                        print('警告：数据文件不存在')
                except Exception as e:
                    print(f'读取数据文件时出错: {e}')
            else:
                print(f'数据存储在Redis中 (host: {CONFIG["redis_host"]}, db: {CONFIG["redis_db"]})')
                try:
                    data_count = self.redis_cli.llen('train_data_buffer')
                    iters = self.redis_cli.get('iters')
                    print(f'  Redis中的数据样本数: {data_count}')
                    print(f'  总对局数: {iters}')
                except:
                    pass
            print('\n程序已安全退出')


collecting_pipeline = CollectPipeline(init_model='current_policy.model')
collecting_pipeline.run()

if CONFIG['use_frame'] == 'paddle':
    collecting_pipeline = CollectPipeline(init_model='current_policy.model')
    collecting_pipeline.run()
elif CONFIG['use_frame'] == 'pytorch':
    collecting_pipeline = CollectPipeline(init_model='current_policy.pkl')
    collecting_pipeline.run()
else:
    print('暂不支持您选择的框架')
    print('训练结束')
