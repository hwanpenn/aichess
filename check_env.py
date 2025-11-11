"""检查环境配置脚本"""
import sys

print("=" * 50)
print("环境检查")
print("=" * 50)

# 检查 Python 版本
print(f"\n1. Python 版本: {sys.version}")
print(f"   Python 路径: {sys.executable}")

# 检查 PyTorch
print("\n2. PyTorch 检查:")
try:
    import torch
    print(f"   ✓ PyTorch 已安装，版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("   ⚠ 警告: CUDA 不可用，请安装 GPU 版本的 PyTorch")
except ImportError:
    print("   ✗ PyTorch 未安装")

# 检查 PaddlePaddle
print("\n3. PaddlePaddle 检查:")
try:
    import paddle
    print(f"   ✓ PaddlePaddle 已安装，版本: {paddle.__version__}")
    print(f"   CUDA 编译支持: {paddle.device.is_compiled_with_cuda()}")
    if paddle.device.is_compiled_with_cuda():
        print(f"   CUDA 版本: {paddle.version.cuda()}")
except ImportError:
    print("   ✗ PaddlePaddle 未安装")

# 检查 NumPy
print("\n4. NumPy 检查:")
try:
    import numpy
    print(f"   ✓ NumPy 已安装，版本: {numpy.__version__}")
except ImportError:
    print("   ✗ NumPy 未安装")

# 检查 Pygame
print("\n5. Pygame 检查:")
try:
    import pygame
    print(f"   ✓ Pygame 已安装，版本: {pygame.version.ver}")
except ImportError:
    print("   ✗ Pygame 未安装")

# 检查 Redis（可选）
print("\n6. Redis 检查（可选）:")
try:
    import redis
    print(f"   ✓ Redis 客户端已安装")
except ImportError:
    print("   - Redis 客户端未安装（可选，用于分布式训练）")

print("\n" + "=" * 50)
print("检查完成")
print("=" * 50)

