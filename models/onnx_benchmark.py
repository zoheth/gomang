import torch
import onnx
import onnxruntime
import time
import numpy as np
import argparse
from tqdm import tqdm
import os
import sys
import platform
import subprocess

def check_gpu_status():
    """详细检查GPU状态并打印诊断信息"""
    print("\n===== GPU 诊断信息 =====")
    
    # 1. 检查PyTorch是否能识别CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"PyTorch检测到CUDA: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
        print(f"当前设备: {torch.cuda.current_device()}")
    else:
        print("PyTorch未检测到CUDA设备")
    
    # 2. 检查ONNX Runtime可用的执行提供程序
    print("\nONNX Runtime版本:", onnxruntime.__version__)
    print("ONNX Runtime可用提供程序:", onnxruntime.get_available_providers())
    
    # 3. 系统信息
    print(f"\n操作系统: {platform.system()} {platform.version()}")
    print(f"Python版本: {sys.version.split()[0]}")
    
    # 4. 检查CUDA是否在系统路径中
    cuda_path = os.environ.get('CUDA_PATH', '')
    print(f"\nCUDA_PATH环境变量: {cuda_path or '未设置'}")
    
    # 5. 尝试运行nvidia-smi (仅限Linux/Windows)
    if platform.system() in ['Linux', 'Windows']:
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                print("\nnvidia-smi 可用:")
                # 只打印前几行
                output_lines = result.stdout.split('\n')
                for i in range(min(10, len(output_lines))):
                    print(output_lines[i])
                if len(output_lines) > 10:
                    print("...")
            else:
                print("\nnvidia-smi 不可用或返回错误:", result.stderr)
        except FileNotFoundError:
            print("\nnvidia-smi 未找到，这表明NVIDIA驱动可能未正确安装")
        except Exception as e:
            print(f"\n运行nvidia-smi时出错: {e}")
    
    # 6. 检查常见问题
    print("\n===== 常见问题检查 =====")
    
    # 检查CUDA版本与PyTorch是否兼容
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"当前CUDA版本 {cuda_version} 与PyTorch兼容")
    else:
        print("可能的问题:")
        print("1. NVIDIA驱动未安装或过期")
        print("2. CUDA工具包未安装或版本不兼容")
        print("3. 安装了CPU版本的PyTorch (而非GPU版本)")
        print("4. 环境变量配置问题")
        
        # 提供安装建议
        print("\n解决方案建议:")
        print("1. 确认您安装了GPU版本的PyTorch:")
        print("   pip list | grep torch")
        print("2. 重新安装GPU版本:")
        print("   pip uninstall torch")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121 # 根据您的CUDA版本调整")
        print("3. 检查GPU驱动:")
        print("   - Windows: 设备管理器 -> 显示适配器")
        print("   - Linux: sudo lshw -C display")
        print("4. 安装最新NVIDIA驱动")

def parse_args():
    parser = argparse.ArgumentParser(description='ONNX模型推理性能测试')
    parser.add_argument('--model', type=str, required=True, help='ONNX模型路径')
    parser.add_argument('--input_shape', type=str, required=True, help='输入形状，例如：1,3,224,224')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='运行设备')
    parser.add_argument('--warmup', type=int, default=10, help='预热次数')
    parser.add_argument('--iterations', type=int, default=100, help='测试迭代次数')
    parser.add_argument('--debug', action='store_true', help='启用详细GPU调试信息')
    return parser.parse_args()

def benchmark_onnx_model(model_path, input_shape, device='cuda', warmup=10, iterations=100, debug=False):
    # 先检查GPU状态（如果需要调试）
    if debug or device == 'cuda':
        check_gpu_status()
    
    # 检查设备可用性
    if device == 'cuda' and not torch.cuda.is_available():
        print("\n警告: CUDA不可用，切换到CPU")
        device = 'cpu'
    
    # 加载ONNX模型
    try:
        onnx_model = onnx.load(model_path)
        print(f"\n模型: {model_path}")
        print(f"图形IR版本: {onnx_model.ir_version}")
        if hasattr(onnx_model, 'producer_name') and onnx_model.producer_name:
            print(f"生产者名称: {onnx_model.producer_name}")
        if hasattr(onnx_model, 'producer_version') and onnx_model.producer_version:
            print(f"生产者版本: {onnx_model.producer_version}")
    except Exception as e:
        print(f"加载ONNX模型元数据时出错: {e}")
        print("继续执行性能测试...")
    
    # 解析输入形状
    input_shape = [int(dim) for dim in input_shape.split(',')]
    print(f"输入形状: {input_shape}")
    
    # 创建随机输入张量
    input_tensor = torch.randn(*input_shape, dtype=torch.float32)
    if device == 'cuda':
        try:
            input_tensor = input_tensor.cuda()
            print("成功将输入张量移至CUDA设备")
        except Exception as e:
            print(f"将张量移至CUDA设备时出错: {e}")
            print("回退到CPU")
            device = 'cpu'
    
    # 选择合适的提供程序
    available_providers = onnxruntime.get_available_providers()
    
    if device == 'cuda' and 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider']
        print("使用CUDA执行提供程序")
    else:
        providers = ['CPUExecutionProvider']
        print("使用CPU执行提供程序")
    
    # 创建ONNX运行时会话
    try:
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if debug:
            # 启用详细日志
            session_options.log_severity_level = 0  # 0: Verbose
            
        session = onnxruntime.InferenceSession(
            model_path, 
            sess_options=session_options,
            providers=providers
        )
        print("成功创建ONNX推理会话")
    except Exception as e:
        print(f"创建ONNX会话时出错: {e}")
        return
    
    # 获取输入名称
    input_name = session.get_inputs()[0].name
    print(f"模型输入名称: {input_name}")
    
    # 准备输入数据
    if device == 'cuda':
        # 对于CUDA，将PyTorch张量转换为NumPy数组
        input_data = input_tensor.cpu().numpy()
    else:
        input_data = input_tensor.numpy()
    
    # 预热
    print(f"\n预热中 ({warmup} 次迭代)...")
    for i in range(warmup):
        try:
            _ = session.run(None, {input_name: input_data})
            if i == 0:
                print("第一次推理成功")
        except Exception as e:
            print(f"预热时出错: {e}")
            return
    
    # 测试性能
    print(f"\n测试中 ({iterations} 次迭代)...")
    latencies = []
    
    for _ in tqdm(range(iterations)):
        # 同步GPU以确保准确计时(仅针对CUDA)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        _ = session.run(None, {input_name: input_data})
        
        # 同步GPU以确保完成(仅针对CUDA)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    # 计算统计数据
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    # 输出性能统计
    print("\n性能统计 (毫秒):")
    print(f"平均延迟: {avg_latency:.4f}")
    print(f"最小延迟: {min_latency:.4f}")
    print(f"最大延迟: {max_latency:.4f}")
    print(f"中位延迟: {median_latency:.4f}")
    print(f"P95延迟: {p95_latency:.4f}")
    print(f"P99延迟: {p99_latency:.4f}")
    
    # 计算每秒推理次数 (FPS)
    fps = 1000 / avg_latency
    print(f"\nFPS: {fps:.2f}")
    
    # 输出内存使用情况
    print("\n内存使用情况:")
    # CPU内存
    process = psutil.Process(os.getpid())
    print(f"CPU内存: {process.memory_info().rss / (1024**2):.2f} MB")
    
    # GPU内存 (仅适用于CUDA)
    if device == 'cuda' and torch.cuda.is_available():
        print("\nGPU内存使用情况:")
        print(f"分配的内存: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        print(f"缓存的内存: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
    
    # 编写结果到文件
    result_file = f"benchmark_results_{os.path.basename(model_path)}_{device}.txt"
    try:
        with open(result_file, 'w') as f:
            f.write(f"模型: {model_path}\n")
            f.write(f"设备: {device}\n")
            f.write(f"输入形状: {input_shape}\n")
            f.write(f"迭代次数: {iterations}\n\n")
            f.write("性能统计 (毫秒):\n")
            f.write(f"平均延迟: {avg_latency:.4f}\n")
            f.write(f"最小延迟: {min_latency:.4f}\n")
            f.write(f"最大延迟: {max_latency:.4f}\n")
            f.write(f"中位延迟: {median_latency:.4f}\n")
            f.write(f"P95延迟: {p95_latency:.4f}\n")
            f.write(f"P99延迟: {p99_latency:.4f}\n\n")
            f.write(f"FPS: {fps:.2f}\n")
        print(f"\n结果已保存到: {result_file}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

def main():
    args = parse_args()
    benchmark_onnx_model(
        args.model, 
        args.input_shape, 
        args.device, 
        args.warmup, 
        args.iterations,
        args.debug
    )

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("警告: psutil模块未安装，内存使用统计将不可用")
        print("可以通过运行 'pip install psutil' 来安装")
    
    try:
        main()
    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
        # 如果出现错误，尝试提供更详细的调试信息
        check_gpu_status()