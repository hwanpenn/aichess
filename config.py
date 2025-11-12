CONFIG = {
    'kill_action': 30,      #和棋回合数
    'dirichlet': 0.2,       # 国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'play_out': 1500,        # 每次移动的模拟次数（RTX 3090优化：批量推理后可增加到1500，提升棋力）
    'c_puct': 5,             # u的权重
    'buffer_size': 200000,   # 经验池大小（64GB内存可支持更大缓冲区）
    'mcts_batch_size': 48,  # MCTS批量推理批次大小（RTX 3090 24GB显存优化：48-64，充分利用GPU）
    'paddle_model_path': 'models/current_policy.model',      # paddle模型路径
    'pytorch_model_path': 'models/current_policy.pkl',   # pytorch模型路径
    'data_dir': 'data',  # 数据存储目录
    'train_data_buffer_path': 'data/train_data_buffer.pkl',   # 数据容器的路径
    'batch_size': 2048,  # 每次更新的train_step数量（RTX 3090 24GB优化：1536-2048，充分利用显存）
    'kl_targ': 0.02,  # kl散度控制
    'epochs' : 15,  # 每次更新的train_step数量（200局样本优化：增加到15-20轮，充分利用小数据集）
    'game_batch_num': 100,  # 训练更新的次数（200局样本优化：100次足够，避免无效循环）
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
    'train_update_interval': 0,  #模型更新间隔时间（秒）（固定数据集优化：设为0，立即训练）
    'use_redis': False, # 数据存储方式
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
    'show_selfplay': False,  # 是否显示自动对弈过程（True=显示，False=不显示，可节省时间）
    'show_delay': 0,  # 每步显示延迟（秒），show_selfplay=True时有效，建议0.3，0=无延迟
    'player2': 'Human',  # Human or MCTS
}