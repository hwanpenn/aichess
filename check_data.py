"""检查训练数据量并给出配置建议"""
import pickle
import os
from config import CONFIG

def check_data():
    """检查数据量并给出建议"""
    data_path = CONFIG['train_data_buffer_path']
    
    if not os.path.exists(data_path):
        print(f'❌ 数据文件不存在: {data_path}')
        print('请先运行 collect.py 收集对弈数据')
        return
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            data_buffer = data.get('data_buffer', [])
            iters = data.get('iters', 0)
            
            data_count = len(data_buffer)
            current_batch_size = CONFIG['batch_size']
            
            print('=' * 60)
            print('📊 训练数据统计')
            print('=' * 60)
            print(f'总对局数: {iters}')
            print(f'训练样本数: {data_count}')
            print(f'当前 batch_size: {current_batch_size}')
            print('=' * 60)
            
            if data_count == 0:
                print('❌ 没有训练数据，请先运行 collect.py 收集数据')
                return
            
            if data_count < current_batch_size:
                print(f'⚠️  警告：数据量不足！')
                print(f'   当前样本数 ({data_count}) < batch_size ({current_batch_size})')
                print(f'   建议将 batch_size 调整为: {min(512, data_count - 1)}')
                print(f'   或者继续收集更多数据（推荐至少 {current_batch_size + 1} 个样本）')
            else:
                # 计算可以训练的次数
                train_times = data_count // current_batch_size
                print(f'✅ 数据量充足，可以训练约 {train_times} 次')
                print(f'   建议的 batch_size 范围: {current_batch_size} - {min(2048, data_count // 2)}')
            
            print('=' * 60)
            
    except Exception as e:
        print(f'❌ 读取数据文件失败: {e}')

if __name__ == '__main__':
    check_data()
