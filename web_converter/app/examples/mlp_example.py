import torch
import torch.nn as nn
import onnx
import os
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

def export_onnx(model, dummy_input, onnx_path):
    """Export the model to ONNX format"""
    model.eval()
    print(f"Exporting model to {onnx_path}...")
    
    kwargs = {
        'export_params': True,
        'opset_version': 11,
        'do_constant_folding': True,
        'input_names': ['input'],
        'output_names': ['output'],
        'dynamic_axes': {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
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
    
    # Verify the exported model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"Successfully exported {onnx_path}")

def train_and_export():
    # Create model
    model = MLPModel()
    
    # Generate some dummy training data
    # In a real scenario, you would use your own dataset
    print("Generating dummy training data...")
    inputs = torch.randn(100, 10)
    targets = torch.randn(100, 2)
    
    # Simple training loop (optional, just for demonstration)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Starting training...")
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 2 == 0:
            print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
    
    # Export to ONNX
    dummy_input = torch.randn(1, 10)
    export_onnx(model, dummy_input, "mlp_model.onnx")

if __name__ == "__main__":
    train_and_export()
