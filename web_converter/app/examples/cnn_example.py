import torch
import torch.nn as nn
import onnx
import os
import inspect

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

def fix_onnx_maxpool(onnx_path):
    """Fix MaxPool and Reshape attributes for better compatibility with onnx2c"""
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

def export_onnx(model, dummy_input, onnx_path):
    """Export the model to ONNX format"""
    model.eval()
    print(f"Exporting model to {onnx_path}...")
    
    kwargs = {
        'export_params': True,
        'opset_version': 11,
        'do_constant_folding': True,
        'input_names': ['input'],
        'output_names': ['output']
    }

    # Add external_data=False if supported by the torch version
    sig = inspect.signature(torch.onnx.export)
    if 'external_data' in sig.parameters:
        kwargs['external_data'] = False
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        **kwargs
    )
    
    # Fix attributes for onnx2c compatibility
    fix_onnx_maxpool(onnx_path)
    
    # Verify the exported model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"Successfully exported {onnx_path}")

def train_and_export():
    # Create model
    model = CNNModel()
    
    # Generate some dummy training data (MNIST-like)
    print("Generating dummy training data...")
    inputs = torch.randn(10, 1, 28, 28)
    targets = torch.randn(10, 10)
    
    # Simple training loop
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Starting training...")
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')
    
    # Export to ONNX
    dummy_input = torch.randn(1, 1, 28, 28)
    export_onnx(model, dummy_input, "cnn_model.onnx")

if __name__ == "__main__":
    train_and_export()
