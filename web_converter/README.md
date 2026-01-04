# ONNX2C Web Converter

一个基于 Docker 的 Web 应用，用于将 ONNX 模型转换为 C 代码，并进行模型验证。

## 功能特性

- 🔄 **ONNX 到 C 转换**: 上传 ONNX 文件，自动转换为优化的 C 代码
- ✅ **模型验证**: 使用 100 组随机数据验证 ONNX 和 C 模型的一致性
- 📊 **详细报告**: 提供平均相对误差、最大相对误差、MAE、MSE 等指标
- 🎨 **现代界面**: 苹果风格的现代化 Web 界面
- 📥 **文件下载**: 转换后的 C 文件可直接下载
- 🐳 **Docker 支持**: 完整的容器化部署

## 快速开始

### 使用 Docker Compose (推荐)

```bash
# 克隆项目
git clone <repository-url>
cd onnx2c/web_converter

# 启动服务
docker-compose up --build
```

### 使用脚本启动

**Linux/macOS:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```cmd
run.bat
```

### 手动 Docker 构建

```bash
# 构建镜像
docker build -t onnx2c-web-converter . -f .\web_converter\Dockerfile        

# 运行容器
docker run -p 5000:5000 \
           --name onnx2c-converter \
           --rm \
           -v "$(pwd)/app/uploads:/app/web_app/uploads" \
           -v "$(pwd)/app/generated:/app/web_app/generated" \
           onnx2c-web-converter
```

## 使用方法

1. 在浏览器中打开 `http://localhost:5000`
2. 拖拽或点击上传 ONNX 模型文件
3. 等待转换和验证完成
4. 查看验证报告和模型对比指标
5. 下载生成的 C 文件

## 支持的指标

应用会生成以下模型对比指标：

- **平均相对误差 (Average Relative Error)**: 平均相对偏差
- **最大相对误差 (Maximum Relative Error)**: 最大相对偏差
- **平均绝对误差 (MAE)**: 平均绝对偏差
- **最大绝对误差 (Maximum Absolute Error)**: 最大绝对偏差
- **均方误差 (MSE)**: 均方根误差
- **测试样本数量**: 用于验证的随机样本数

## 项目结构

```
web_converter/
├── Dockerfile              # Docker 镜像定义
├── docker-compose.yml      # Docker Compose 配置
├── run.sh                  # Linux/macOS 启动脚本
├── run.bat                 # Windows 启动脚本
├── README.md              # 项目说明
└── app/                   # Web 应用代码
    ├── app.py             # Flask 主应用
    ├── templates/         # HTML 模板
    │   └── index.html     # 主页面
    ├── uploads/           # 上传文件存储
    └── generated/         # 生成文件存储
```

## 技术栈

- **后端**: Python Flask
- **前端**: HTML/CSS/JavaScript (苹果设计风格)
- **模型处理**: ONNX, ONNXRuntime
- **编译**: GCC
- **容器化**: Docker

## 环境要求

- Docker
- Docker Compose (可选)

## 开发说明

### 本地开发

如果需要在本地开发环境中运行:

```bash
# 安装依赖
pip install flask werkzeug onnx numpy onnxruntime

# 设置环境变量
export ONNX2C_PATH=/path/to/onnx2c/build/onnx2c

# 运行应用
cd app
python app.py
```

### 自定义配置

可以通过环境变量自定义配置：

- `ONNX2C_PATH`: onnx2c 可执行文件路径
- `PYTHONPATH`: Python 模块路径

## 故障排除

### 常见问题

1. **构建失败**: 确保安装了 Docker 和必要的依赖
2. **转换错误**: 检查 ONNX 文件格式和大小限制 (100MB)
3. **编译失败**: 确保生成的 C 代码语法正确

### 日志查看

```bash
# 查看容器日志
docker logs onnx2c-converter

# 实时监控日志
docker logs -f onnx2c-converter
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目遵循与 onnx2c 主项目相同的许可证。