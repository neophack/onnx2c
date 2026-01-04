import torch
import torch.nn as nn
import onnx
import subprocess
import os
import numpy as np
import inspect

# 1. Fully Connected Model (MLP)
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Convolutional Model (CNN)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 14 * 14, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc(x)
        return x

# 3. LSTM Model - 修复版：只返回最后一个时间步的输出
class LSTMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        # 初始化权重为1.0，参考 lstm.py
        nn.init.constant_(self.lstm.weight_ih_l0, 1.0)
        nn.init.constant_(self.lstm.weight_hh_l0, 1.0)
        nn.init.constant_(self.lstm.bias_ih_l0, 1.0)
        nn.init.constant_(self.lstm.bias_hh_l0, 1.0)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def fix_onnx_maxpool(onnx_path):
    """修复ONNX模型中的MaxPool和Reshape属性"""
    model = onnx.load(onnx_path)
    modified = False
    for node in model.graph.node:
        if node.op_type == 'MaxPool':
            # Remove storage_order attribute
            new_attrs = [a for a in node.attribute if a.name != 'storage_order']
            if len(new_attrs) < len(node.attribute):
                del node.attribute[:]
                node.attribute.extend(new_attrs)
                modified = True
        if node.op_type == 'Reshape':
            # Remove allowzero attribute
            new_attrs = [a for a in node.attribute if a.name != 'allowzero']
            if len(new_attrs) < len(node.attribute):
                del node.attribute[:]
                node.attribute.extend(new_attrs)
                modified = True
    if modified:
        onnx.save(model, onnx_path)
        print(f"Fixed attributes in {onnx_path}")

def export_onnx(model, dummy_input, onnx_path, model_name="model"):
    """导出ONNX模型，针对不同模型类型使用不同配置"""
    model.eval()
    try:
        is_rnn = "LSTM" in model_name or "GRU" in model_name
        
        kwargs = {
            'export_params': True,
            'opset_version': 14 if is_rnn else 11,  # RNN使用更高版本opset以获得更好支持
            'do_constant_folding': True,
            'input_names': ['input'],
            'keep_initializers_as_inputs': False,
        }
        
        # 根据模型类型设置输出名称
        kwargs['output_names'] = ['output']
        print(f"  Exporting {model_name} with single output")

        if is_rnn:
            # 参考 lstm.py 的导出配置
            kwargs['operator_export_type'] = torch.onnx.OperatorExportTypes.ONNX
            kwargs['dynamo'] = False
            kwargs['opset_version'] = 14
            kwargs['dynamic_axes'] = {
                'input': {1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
            # 使用 torch.jit.trace，参考 lstm.py
            print(f"  Tracing model {model_name}...")
            model = torch.jit.trace(model, dummy_input)

        # 检查是否支持external_data参数
        sig = inspect.signature(torch.onnx.export)
        if 'external_data' in sig.parameters:
            kwargs['external_data'] = False

        # 导出ONNX
        with torch.no_grad():
            torch.onnx.export(model, dummy_input, onnx_path, **kwargs)
        
        # 验证ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # 修复可能的属性问题
        fix_onnx_maxpool(onnx_path)
        
        print(f"  [OK] Successfully exported to {onnx_path}")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed to export {onnx_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_onnx2c(onnx_path, c_path, onnx2c_bin):
    """运行onnx2c转换器"""
    if not os.path.exists(onnx_path):
        return False, "ONNX file not found"
    try:
        result = subprocess.run(
            [onnx2c_bin, onnx_path],
            capture_output=True,
            text=True,
            check=False,
            encoding='utf-8',
            errors='ignore',
            timeout=30  # 添加超时
        )
        
        if result.returncode == 0:
            with open(c_path, 'w') as f:
                f.write(result.stdout)
            return True, ""
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "onnx2c conversion timeout"
    except Exception as e:
        return False, str(e)

def main():
    onnx2c_bin = r"..\cmake-build-debug\onnx2c.exe"
    if not os.path.exists(onnx2c_bin):
        print(f"Error: onnx2c binary not found at {onnx2c_bin}")
        print("Please build onnx2c first or update the path.")
        return

    # 测试模型列表
    models = [
        ("MLP", MLPModel(), torch.randn(1, 10)),
        ("CNN", CNNModel(), torch.randn(1, 1, 28, 28)),
        ("LSTM-FC", LSTMModel(), torch.randn(1, 5, 10)),
    ]

    results = []

    print("=" * 70)
    print("PyTorch to ONNX to C Conversion Test")
    print("=" * 70)

    for name, model, dummy_input in models:
        print(f"\n--- Testing Model: {name} ---")
        onnx_path = f"{name.lower().replace('-', '_')}.onnx"
        c_path = f"{name.lower().replace('-', '_')}.c"
        
        # 导出ONNX
        exported = export_onnx(model, dummy_input, onnx_path, name)
        
        if exported:
            print(f"  Converting to C...")
            success, error = run_onnx2c(onnx_path, c_path, onnx2c_bin)
            if success:
                print(f"  [OK] Successfully converted to {c_path}")
            else:
                print(f"  [FAIL] onnx2c conversion failed")
            results.append((name, success, error))
        else:
            results.append((name, False, "ONNX export failed"))

    # 打印最终报告
    print("\n\n" + "=" * 70)
    print("Final Report")
    print("=" * 70)
    print(f"{'Model Name':<20} | {'Export':<8} | {'onnx2c':<8} | {'Error'}")
    print("-" * 70)
    
    for name, success, error in results:
        export_status = "PASS" if "failed" not in error.lower() and error != "ONNX export failed" else "FAIL"
        onnx2c_status = "PASS" if success else "FAIL"
        
        err_msg = ""
        if error:
            lines = error.strip().split('\n')
            err_msg = lines[-1][:50] if lines else error[:50]
        
        print(f"{name:<20} | {export_status:<8} | {onnx2c_status:<8} | {err_msg}")

    print("\n" + "=" * 70)
    
    # 统计成功率
    total = len(results)
    passed = sum(1 for _, success, _ in results if success)
    print(f"Success Rate: {passed}/{total} ({100*passed//total}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()