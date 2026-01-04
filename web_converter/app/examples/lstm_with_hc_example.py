import torch
import torch.nn as nn
import onnx
import os
import inspect
import traceback

# 3. LSTM Model with h and c as inputs
class LSTMWithHCModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=1):
        super(LSTMWithHCModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.constant_(param, 1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 1.0)

    def forward(self, x, h, c):
        # x: (batch, seq_len, input_size)
        # h: (num_layers, batch, hidden_size)
        # c: (num_layers, batch, hidden_size)
        out, (h_out, c_out) = self.lstm(x, (h, c))
        
        # Only take the last time step for the main output
        # out: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])
        
        # Return main output and the new states
        return out, h_out, c_out

def export_onnx(model, dummy_inputs, onnx_path):
    """Export the model to ONNX format with LSTM specific configuration"""
    model.eval()
    print(f"Exporting model to {onnx_path}...")
    
    try:
        kwargs = {
            'export_params': True,
            'opset_version': 14,  # Use opset 14 for better RNN support
            'do_constant_folding': True,
            'input_names': ['input', 'h0', 'c0'],
            'output_names': ['output', 'hn', 'cn'],
            'keep_initializers_as_inputs': False,
            'dynamic_axes': {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'h0': {1: 'batch_size'},
                'c0': {1: 'batch_size'},
                'output': {0: 'batch_size'},
                'hn': {1: 'batch_size'},
                'cn': {1: 'batch_size'}
            }
        }
        
        # Check for optional parameters based on torch version
        sig = inspect.signature(torch.onnx.export)
        if 'external_data' in sig.parameters:
            kwargs['external_data'] = False
        if 'dynamo' in sig.parameters:
            kwargs['dynamo'] = False

        # Use torch.jit.trace for RNN models to ensure consistent graph
        print("Tracing model...")
        traced_model = torch.jit.trace(model, dummy_inputs)

        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(traced_model, dummy_inputs, onnx_path, **kwargs)
        
        # Verify the exported model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"Successfully exported {onnx_path}")
        return True
    except Exception as e:
        print(f"Failed to export {onnx_path}: {e}")
        traceback.print_exc()
        return False

def run_example():
    # Parameters
    input_size = 10
    hidden_size = 20
    num_layers = 1
    batch_size = 1
    seq_length = 5
    
    # Create model
    model = LSTMWithHCModel(input_size, hidden_size, num_layers)
    
    # Prepare dummy inputs
    x = torch.randn(batch_size, seq_length, input_size)
    h = torch.randn(num_layers, batch_size, hidden_size)
    c = torch.randn(num_layers, batch_size, hidden_size)
    dummy_inputs = (x, h, c)
    
    # Export path
    onnx_path = "lstm_with_hc.onnx"
    
    # Export
    success = export_onnx(model, dummy_inputs, onnx_path)
    
    if success:
        print(f"Model exported to {os.path.abspath(onnx_path)}")
        
        # Print input/output info
        onnx_model = onnx.load(onnx_path)
        print("\nONNX Model Inputs:")
        for input in onnx_model.graph.input:
            print(f"  Name: {input.name}")
        
        print("\nONNX Model Outputs:")
        for output in onnx_model.graph.output:
            print(f"  Name: {output.name}")

if __name__ == "__main__":
    run_example()
