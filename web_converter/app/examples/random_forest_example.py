import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

# 随机森林示例 (Random Forest)
def train_and_export():
    # 1. 生成模拟分类数据
    print("Generating dummy training data...")
    # 使用 10 个特征，与 mlp_example 一致
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=2, random_state=42)
    X = X.astype(np.float32)

    # 2. 创建并训练随机森林模型
    print("Training Random Forest model...")
    # 限制树的数量和深度以保持 ONNX 模型较小，便于演示
    clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    clf.fit(X, y)

    # 3. 转换为 ONNX 格式
    print("Converting model to ONNX...")
    # 定义输入类型：[None, 10] 表示 batch_size 可变，特征数为 10
    initial_type = [('input', FloatTensorType([None, 10]))]
    
    # options={type(clf): {'zipmap': False}} 很重要
    # 许多嵌入式或轻量级 ONNX 运行时（如 onnx2c 可能的目标环境）不支持 ZipMap 节点
    options = {type(clf): {'zipmap': False}}
    
    onnx_model = convert_sklearn(
        clf, 
        initial_types=initial_type, 
        options=options, 
        target_opset=12
    )

    # 4. 保存模型
    onnx_path = os.path.join(os.path.dirname(__file__), "random_forest_model.onnx")
    onnx.save(onnx_model, onnx_path)
    
    # 验证导出的模型
    onnx.checker.check_model(onnx_model)
    print(f"Successfully exported {onnx_path}")

if __name__ == "__main__":
    train_and_export()
