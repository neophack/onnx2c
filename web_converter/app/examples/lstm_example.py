import torch
import torch.nn as nn
import onnx
import os
import inspect
import traceback

# 3. LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        nn.init.constant_(self.lstm.weight_ih_l0, 1.0)
        nn.init.constant_(self.lstm.weight_hh_l0, 1.0)
        nn.init.constant_(self.lstm.bias_ih_l0, 1.0)
        nn.init.constant_(self.lstm.bias_hh_l0, 1.0)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Only take the last time step
        out = self.fc(out[:, -1, :])
        return out

def export_onnx(model, dummy_input, onnx_path):
    """Export the model to ONNX format with LSTM specific configuration"""
    model.eval()
    print(f"Exporting model to {onnx_path}...")
    
    try:
        kwargs = {
            'export_params': True,
            'opset_version': 14,  # Use opset 14 for better RNN support
            'do_constant_folding': True,
            'input_names': ['input'],
            'output_names': ['output'],
            'keep_initializers_as_inputs': False,
            'operator_export_type': torch.onnx.OperatorExportTypes.ONNX,
            'dynamic_axes': {
                'input': {1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        }
        
        # Use torch.jit.trace for RNN models to ensure consistent graph
        print("Tracing model...")
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Check for optional parameters based on torch version
        sig = inspect.signature(torch.onnx.export)
        if 'external_data' in sig.parameters:
            kwargs['external_data'] = False
        if 'dynamo' in sig.parameters:
            kwargs['dynamo'] = False

        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(traced_model, dummy_input, onnx_path, **kwargs)
        
        # Verify the exported model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"Successfully exported {onnx_path}")
        return True
    except Exception as e:
        print(f"Failed to export {onnx_path}: {e}")
        traceback.print_exc()
        return False

def train_and_export():
    # Create model
    input_size = 10
    hidden_size = 20
    model = LSTMModel(input_size, hidden_size)
    
    # Generate some dummy training data
    print("Generating dummy training data...")
    batch_size = 8
    seq_length = 5
    inputs = torch.randn(batch_size, seq_length, input_size)
    targets = torch.randn(batch_size, 1)
    
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
    dummy_input = torch.randn(1, seq_length, input_size)
    export_onnx(model, dummy_input, "lstm_model.onnx")

if __name__ == "__main__":
    train_and_export()
