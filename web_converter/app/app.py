import os
import subprocess
import tempfile
import shutil
import numpy as np
import onnx
import onnxruntime as ort
from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import json
from datetime import datetime, timedelta
import math
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GENERATED_FOLDER'] = 'generated'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Ensure upload and generated folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

ONNX2C_PATH = os.environ.get('ONNX2C_PATH', '/usr/bin/onnx2c')

def cleanup_old_files():
    """清理10分钟前的文件"""
    try:
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=10)
        
        folders_to_clean = [app.config['UPLOAD_FOLDER'], app.config['GENERATED_FOLDER']]
        
        for folder in folders_to_clean:
            if not os.path.exists(folder):
                continue
                
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff_time:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"[CLEANUP] Deleted old file: {file_path}")
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            print(f"[CLEANUP] Deleted old directory: {file_path}")
                except Exception as e:
                    print(f"[CLEANUP ERROR] Failed to delete {file_path}: {e}")
    except Exception as e:
        print(f"[CLEANUP ERROR] Cleanup failed: {e}")

def periodic_cleanup():
    """定期执行清理任务"""
    while True:
        try:
            cleanup_old_files()
            # 每5分钟执行一次清理
            time.sleep(300)  # 300秒 = 5分钟
        except Exception as e:
            print(f"[CLEANUP THREAD ERROR] {e}")
            time.sleep(60)  # 出错时等待1分钟后重试

# 启动清理线程
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

# Global dictionary to store task status and logs
tasks = {}
tasks_lock = threading.Lock()

def add_log(task_id, message):
    """Add a log message to a specific task"""
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id]['logs'].append(message)
            # Also print to console for debugging
            print(f"[{task_id}] {message}")

def update_task_status(task_id, status, result=None, error=None):
    """Update the status and result of a specific task"""
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id]['status'] = status
            if result is not None:
                tasks[task_id]['result'] = result
            if error is not None:
                tasks[task_id]['error'] = error

def background_conversion(task_id, upload_path, filename, conversion_id):
    """Perform model conversion in the background"""
    try:
        execution_logs = []
        def log_msg(msg):
            add_log(task_id, msg)
            execution_logs.append(msg)

        log_msg(f"[START] Starting conversion process for {filename}")
        log_msg(f"[INFO] Conversion ID: {conversion_id}")
        log_msg(f"[INFO] File saved to: {upload_path}")
        
        # Convert to C
        c_output_path = os.path.join(app.config['GENERATED_FOLDER'], f"{conversion_id}_generated.c")
        success, message, conv_logs = run_onnx2c(upload_path, c_output_path)
        for l in conv_logs: log_msg(l)
        
        if not success:
            log_msg(f"[FINAL] Conversion failed: {message}")
            update_task_status(task_id, 'failed', error=f'Conversion failed: {message}', result={'logs': execution_logs})
            return
        
        # Generate main.c
        main_c_path = os.path.join(app.config['GENERATED_FOLDER'], f"{conversion_id}_main.c")
        log_msg(f"[INFO] Generating main.c file: {main_c_path}")
        create_main_c(main_c_path, conversion_id, upload_path)
        log_msg(f"[SUCCESS] Main.c file generated successfully")
        
        # Compile the model
        executable_path = os.path.join(app.config['GENERATED_FOLDER'], f"{conversion_id}_model")
        compile_success, compile_message, compile_logs, mcu_resources = compile_c_model(c_output_path, main_c_path, executable_path)
        for l in compile_logs: log_msg(l)
        
        if not compile_success:
            log_msg(f"[FINAL] Compilation failed: {compile_message}")
            update_task_status(task_id, 'failed', error=f'Compilation failed: {compile_message}', result={'logs': execution_logs})
            return
            
        # Add MCU resources to perf_metrics later
        log_msg(f"[INFO] MCU Resource Estimates: ROM={mcu_resources['rom']} bytes, RAM={mcu_resources['ram']} bytes")
        
        # Get model information
        log_msg("[INFO] Extracting model information...")
        model_info = get_model_info(upload_path)
        if model_info:
            log_msg(f"[SUCCESS] Model info extracted: {model_info['total_nodes']} nodes, {len(model_info['inputs'])} inputs, {len(model_info['outputs'])} outputs")
        else:
            log_msg("[WARNING] Failed to extract model information")
        
        # Generate test data and run validation
        log_msg("[INFO] Generating test data for validation...")
        inputs, input_shapes = generate_random_inputs(upload_path, 10)
        
        validation_report = {
            'status': 'not_run',
            'message': 'Validation not performed',
            'metrics': None,
            'error': None
        }
        
        if inputs and len(inputs) > 0:
            log_msg(f"[SUCCESS] Generated {len(inputs)} test samples")
            
            try:
                # Run ONNX inference
                log_msg(f"[INFO] Running ONNX inference with {len(inputs)} samples...")
                onnx_results, onnx_logs = run_onnx_inference(upload_path, inputs)
                for l in onnx_logs: log_msg(l)
                
                if onnx_results is None or len(onnx_results) == 0:
                    log_msg(f"[ERROR] ONNX inference failed - got {len(onnx_results) if onnx_results else 0} results")
                else:
                    log_msg(f"[SUCCESS] ONNX inference completed, got {len(onnx_results)} results")
                
                # Run C inference
                log_msg(f"[INFO] Running C inference with {len(inputs)} samples...")
                c_results, c_logs, perf_metrics = run_c_inference(executable_path, inputs)
                for l in c_logs: log_msg(l)
                
                # Merge MCU resources into perf_metrics
                if perf_metrics and mcu_resources:
                    perf_metrics['mcu_rom'] = mcu_resources['rom']
                    perf_metrics['mcu_ram'] = mcu_resources['ram']
                
                if onnx_results and c_results:
                    log_msg("[INFO] Calculating validation metrics...")
                    # Calculate metrics
                    metrics = calculate_metrics(onnx_results, c_results, perf_metrics)
                    if metrics:
                        log_msg("[SUCCESS] Validation metrics calculated successfully")
                        log_msg(f"[METRICS] MAE: {metrics['overall_metrics']['mae']:.2e}")
                        log_msg(f"[METRICS] Max Error: {metrics['overall_metrics']['max_absolute_error']:.2e}")
                        log_msg(f"[METRICS] Avg Rel Error: {metrics['overall_metrics']['avg_relative_error']:.2e}")
                        
                        validation_report = {
                            'status': 'success',
                            'message': f'Validation completed with {len(inputs)} test samples',
                            'metrics': metrics,
                            'error': None
                        }
                    else:
                        log_msg("[ERROR] Failed to calculate validation metrics")
                        validation_report = {
                            'status': 'failed',
                            'message': 'Failed to calculate validation metrics',
                            'metrics': None,
                            'error': 'Metrics calculation failed'
                        }
                else:
                    log_msg(f"[ERROR] Inference comparison failed - ONNX: {len(onnx_results)}, C: {len(c_results)}")
                    validation_report = {
                        'status': 'failed',
                        'message': 'Failed to run inference comparison',
                        'metrics': None,
                        'error': f'ONNX results: {len(onnx_results) if onnx_results else 0}, C results: {len(c_results) if c_results else 0}'
                    }
            except Exception as e:
                log_msg(f"[EXCEPTION] Validation error: {str(e)}")
                validation_report = {
                    'status': 'error',
                    'message': f'Validation error: {str(e)}',
                    'metrics': None,
                    'error': str(e)
                }
        else:
            log_msg("[WARNING] No test inputs generated - validation skipped")
            validation_report = {
                'status': 'skipped',
                'message': 'No test inputs generated - validation skipped',
                'metrics': None,
                'error': 'Failed to generate test inputs'
            }
        
        log_msg("[FINAL] Conversion process completed successfully")
        
        # Store final result
        result = {
            'conversion_id': conversion_id,
            'filename': filename,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'execution_logs': execution_logs,
            'validation': validation_report,
            'model_info': model_info,
            'files': {
                'c_file': f"{conversion_id}_generated.c",
                'main_file': f"{conversion_id}_main.c",
                'executable': f"{conversion_id}_model"
            }
        }
        
        # Save result to file for report route
        result_path = os.path.join(app.config['GENERATED_FOLDER'], f"{conversion_id}_result.json")
        try:
            with open(result_path, 'w') as f:
                json.dump(result, f)
            log_msg(f"[INFO] Result saved to {result_path}")
        except Exception as e:
            log_msg(f"[ERROR] Failed to save result to {result_path}: {e}")
            
        update_task_status(task_id, 'completed', result=result)
        
    except Exception as e:
        error_msg = f"Unexpected error during background conversion: {str(e)}"
        print(error_msg)
        update_task_status(task_id, 'failed', error=error_msg)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'onnx'

def run_onnx2c(input_file, output_file):
    """Run onnx2c converter"""
    try:
        log_lines = []
        # Use file object for redirection - more "Pythonic" and just as fast as shell redirection
        # while avoiding shell=True security/portability issues.
        # -l0 suppresses verbose logging to stderr, which was the other bottleneck.
        with open(output_file, 'w') as f_out:
            cmd = [ONNX2C_PATH, input_file, "-l0"]
            log_lines.append(f"[INFO] Running command: {' '.join(cmd)} > {output_file}")
            
            result = subprocess.run(cmd, stdout=f_out, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            log_lines.append(f"[ERROR] Command failed with return code: {result.returncode}")
            if result.stderr:
                log_lines.append(f"[ERROR] Error output: {result.stderr}")
            return False, result.stderr or "Unknown error", log_lines
        
        log_lines.append("[SUCCESS] ONNX to C conversion completed successfully")
        return True, "Success", log_lines
    except Exception as e:
        error_msg = str(e)
        log_lines = [f"[EXCEPTION] Error running onnx2c: {error_msg}"]
        return False, error_msg, log_lines

def compile_c_model(c_file, main_file, executable):
    """Compile C model with main.c and estimate resources"""
    log_lines = []
    mcu_resources = {
        'rom': 0,
        'ram': 0
    }
    try:
        # Compile the generated C file as object file first
        obj_file = c_file.replace('.c', '.o')
        cmd_obj = ['gcc', '-c', '-O2', '-ffast-math', c_file, '-o', obj_file]
        cmd_obj_str = ' '.join(cmd_obj)
        log_lines.append(f"[INFO] Compiling object file: {cmd_obj_str}")
        
        result_obj = subprocess.run(cmd_obj, capture_output=True, text=True)
        
        if result_obj.returncode != 0:
            log_lines.append(f"[ERROR] Object compilation failed with return code: {result_obj.returncode}")
            log_lines.append(f"[ERROR] Compiler output: {result_obj.stderr}")
            if result_obj.stdout:
                log_lines.append(f"[INFO] Compiler stdout: {result_obj.stdout}")
            return False, f"Object compilation failed: {result_obj.stderr}", log_lines, None
        
        log_lines.append("[SUCCESS] Object file compiled successfully")
        
        # Estimate resources using size tool
        try:
            # Use Berkeley format for size output
            cmd_size = ['size', '-A', obj_file]
            result_size = subprocess.run(cmd_size, capture_output=True, text=True)
            if result_size.returncode == 0:
                log_lines.append(f"[INFO] Size output:\n{result_size.stdout}")
                # Parse section sizes
                rom_sections = ['.text', '.rodata']
                ram_sections = ['.data', '.bss']
                
                rom_total = 0
                ram_total = 0
                
                for line in result_size.stdout.split('\n'):
                    parts = line.split()
                    if len(parts) >= 2:
                        section_name = parts[0]
                        try:
                            section_size = int(parts[1])
                            if any(section_name.startswith(s) for s in rom_sections):
                                rom_total += section_size
                            elif any(section_name.startswith(s) for s in ram_sections):
                                ram_total += section_size
                        except ValueError:
                            continue
                
                mcu_resources['rom'] = rom_total
                mcu_resources['ram'] = ram_total
                log_lines.append(f"[MCU] Estimated ROM (Code + Read-only data): {rom_total} bytes")
                log_lines.append(f"[MCU] Estimated RAM (Data + BSS): {ram_total} bytes")
        except Exception as e:
            log_lines.append(f"[WARNING] Could not estimate MCU resources: {e}")

        # Then compile main.c and link with object file
        cmd_link = ['gcc', '-O2', '-ffast-math', main_file, obj_file, '-o', executable, '-lm']
        cmd_link_str = ' '.join(cmd_link)
        log_lines.append(f"[INFO] Linking executable: {cmd_link_str}")
        
        result_link = subprocess.run(cmd_link, capture_output=True, text=True)
        
        if result_link.returncode != 0:
            log_lines.append(f"[ERROR] Linking failed with return code: {result_link.returncode}")
            log_lines.append(f"[ERROR] Linker output: {result_link.stderr}")
            return False, f"Linking failed: {result_link.stderr}", log_lines, None
        
        log_lines.append("[SUCCESS] Executable linked successfully")
        return True, "Success", log_lines, mcu_resources
    except Exception as e:
        error_msg = str(e)
        log_lines.append(f"[EXCEPTION] Error during compilation: {error_msg}")
        return False, error_msg, log_lines, None

def generate_random_inputs(model_path, num_samples=10):
    """Generate random inputs based on ONNX model input shapes"""
    try:
        model = onnx.load(model_path)
        inputs = []
        input_shapes = []
        
        # Parse all input tensors
        for input_tensor in model.graph.input:
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)  # Default for dynamic shapes
            input_shapes.append({
                'name': input_tensor.name,
                'shape': shape,
                'dtype': 'float32'
            })
        
        # Generate random data for each sample
        for _ in range(num_samples):
            sample_inputs = []
            for input_info in input_shapes:
                if input_info['dtype'] == 'float32':
                    # Generate normalized random data
                    data = np.random.uniform(-1.0, 1.0, input_info['shape']).astype(np.float32)
                else:
                    data = np.random.randn(*input_info['shape']).astype(np.float32)
                sample_inputs.append(data)
            inputs.append(sample_inputs)
        
        return inputs, input_shapes
    except Exception as e:
        print(f"Error generating inputs: {e}")
        return [], []

def run_onnx_inference(model_path, inputs):
    """Run inference with ONNX model"""
    log_lines = []
    try:
        # Create ONNX Runtime session
        log_lines.append(f"[DEBUG] Creating ONNX Runtime session for {model_path}")
        session = ort.InferenceSession(model_path)
        input_names = [inp.name for inp in session.get_inputs()]
        
        log_lines.append(f"[DEBUG] ONNX model input names: {input_names}")
        log_lines.append(f"[DEBUG] Number of input samples: {len(inputs)}")
        
        results = []
        for i, sample_inputs in enumerate(inputs):
            log_lines.append(f"[DEBUG] Processing ONNX sample {i+1}/{len(inputs)}")
            
            # Create input dictionary
            input_dict = {}
            for j, input_name in enumerate(input_names):
                if j < len(sample_inputs):
                    input_dict[input_name] = sample_inputs[j]
                    log_lines.append(f"[DEBUG] Input '{input_name}' shape: {sample_inputs[j].shape}")
                else:
                    log_lines.append(f"[WARNING] Missing input for '{input_name}' in sample {i+1}")
            
            # Run inference
            output = session.run(None, input_dict)
            results.append(output)
            log_lines.append(f"[DEBUG] Sample {i+1} output shapes: {[out.shape for out in output]}")
        
        log_lines.append(f"[SUCCESS] ONNX inference completed with {len(results)} results")
        return results, log_lines
    except Exception as e:
        log_lines.append(f"[ERROR] ONNX inference failed: {e}")
        import traceback
        log_lines.append(f"[ERROR] Traceback: {traceback.format_exc()}")
        return [], log_lines

def run_c_inference(executable, inputs):
    """Run inference with compiled C model"""
    log_lines = []
    try:
        results = []
        inference_times = []
        memory_usages = []
        log_lines.append(f"[INFO] Starting C inference with {len(inputs)} samples")
        log_lines.append(f"[INFO] Executable: {executable}")
        
        for i, sample_inputs in enumerate(inputs):
            log_lines.append(f"[INFO] Processing sample {i+1}/{len(inputs)}")
            
            # Create temporary input file with all inputs concatenated
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                total_elements = 0
                for input_data in sample_inputs:
                    np.savetxt(f, input_data.flatten(), fmt='%.15g')
                    total_elements += input_data.size
                input_file = f.name
                log_lines.append(f"[INFO] Created input file {input_file} with {total_elements} elements")
            
            try:
                # Run C executable
                cmd = [executable, input_file]
                cmd_str = ' '.join(cmd)
                log_lines.append(f"[INFO] Running: {cmd_str}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    log_lines.append(f"[SUCCESS] Sample {i+1} executed successfully")
                    
                    # Parse metrics from stderr
                    stderr_content = result.stderr
                    for line in stderr_content.split('\n'):
                        if '[METRICS] time:' in line:
                            try:
                                t = float(line.split('time:')[1].split('s')[0].strip())
                                inference_times.append(t)
                            except: pass
                        if '[METRICS] memory:' in line:
                            try:
                                m = int(line.split('memory:')[1].split('KB')[0].strip())
                                memory_usages.append(m)
                            except: pass

                    # Parse output
                    output_lines = result.stdout.strip().split('\n')
                    output_data = []
                    valid_lines = 0
                    for line in output_lines:
                        if line.strip():
                            try:
                                output_data.append(float(line))
                                valid_lines += 1
                            except ValueError:
                                continue
                    
                    log_lines.append(f"[INFO] Parsed {valid_lines} output values from {len(output_lines)} lines")
                    
                    if output_data:
                        results.append([np.array(output_data)])
                    else:
                        log_lines.append("[WARNING] No valid output data found, using zero")
                        results.append([np.array([0.0])])
                else:
                    log_lines.append(f"[ERROR] Sample {i+1} failed with return code: {result.returncode}")
                    log_lines.append(f"[ERROR] stderr: {result.stderr}")
                    log_lines.append(f"[ERROR] stdout: {result.stdout}")
                    results.append([np.array([0.0])])
                    
            finally:
                if os.path.exists(input_file):
                    os.unlink(input_file)
                    log_lines.append(f"[INFO] Cleaned up input file {input_file}")
        
        log_lines.append(f"[SUCCESS] C inference completed for all {len(inputs)} samples")
        return results, log_lines, {
            'times': inference_times,
            'memory': memory_usages
        }
    except Exception as e:
        error_msg = str(e)
        log_lines.append(f"[EXCEPTION] Error in C inference: {error_msg}")
        return [], log_lines, None

def get_model_info(model_path):
    """Get detailed model information"""
    try:
        model = onnx.load(model_path)
        
        # Get input information
        inputs = []
        for inp in model.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Dynamic dimension
            inputs.append({
                'name': inp.name,
                'shape': shape,
                'type': inp.type.tensor_type.elem_type
            })
        
        # Get output information
        outputs = []
        for out in model.graph.output:
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Dynamic dimension
            outputs.append({
                'name': out.name,
                'shape': shape,
                'type': out.type.tensor_type.elem_type
            })
        
        # Get model metadata
        model_info = {
            'producer_name': model.producer_name,
            'producer_version': model.producer_version,
            'domain': model.domain,
            'model_version': model.model_version,
            'doc_string': model.doc_string,
            'ir_version': model.ir_version,
            'opset_version': model.opset_import[0].version if model.opset_import else 'unknown'
        }
        
        # Count nodes by type
        node_counts = {}
        for node in model.graph.node:
            op_type = node.op_type
            node_counts[op_type] = node_counts.get(op_type, 0) + 1
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'model_info': model_info,
            'node_counts': node_counts,
            'total_nodes': len(model.graph.node)
        }
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None

def calculate_metrics(onnx_results, c_results, perf_metrics=None):
    """Calculate comparison metrics and performance metrics"""
    if not onnx_results or not c_results or len(onnx_results) != len(c_results):
        return None
    
    try:
        relative_errors = []
        absolute_errors = []
        sample_errors = []
        
        for i, (onnx_out, c_out) in enumerate(zip(onnx_results, c_results)):
            sample_error = []
            
            # Handle multiple outputs
            if isinstance(onnx_out, list):
                onnx_outputs = onnx_out
            else:
                onnx_outputs = [onnx_out]
                
            if isinstance(c_out, list):
                c_outputs = c_out
            else:
                c_outputs = [c_out]
            
            for j, (onnx_single, c_single) in enumerate(zip(onnx_outputs, c_outputs)):
                onnx_flat = onnx_single.flatten() if hasattr(onnx_single, 'flatten') else np.array([onnx_single]).flatten()
                c_flat = c_single.flatten() if hasattr(c_single, 'flatten') else np.array([c_single]).flatten()
                
                # Ensure same length
                min_len = min(len(onnx_flat), len(c_flat))
                onnx_flat = onnx_flat[:min_len]
                c_flat = c_flat[:min_len]
                
                if min_len > 0:
                    # Calculate absolute error
                    abs_error = np.abs(onnx_flat - c_flat)
                    absolute_errors.extend(abs_error)
                    
                    # Calculate relative error (avoid division by zero)
                    rel_error = np.where(np.abs(onnx_flat) > 1e-10, 
                                       abs_error / np.abs(onnx_flat), 
                                       abs_error)
                    relative_errors.extend(rel_error)
                    
                    # Store per-sample error
                    sample_error.append({
                        'output_index': j,
                        'mae': float(np.mean(abs_error)),
                        'max_error': float(np.max(abs_error)),
                        'avg_rel_error': float(np.mean(rel_error))
                    })
            
            sample_errors.append(sample_error)
        
        if len(absolute_errors) == 0:
            return None
            
        absolute_errors = np.array(absolute_errors)
        relative_errors = np.array(relative_errors)
        
        # Calculate overall metrics
        mae = float(np.mean(absolute_errors))
        max_abs_error = float(np.max(absolute_errors))
        avg_rel_error = float(np.mean(relative_errors))
        max_rel_error = float(np.max(relative_errors))
        mse = float(np.mean(absolute_errors ** 2))
        rmse = float(np.sqrt(mse))
        
        # Performance metrics
        avg_time = 0
        max_time = 0
        mcu_rom = 0
        mcu_ram = 0
        if perf_metrics:
            if perf_metrics['times']:
                avg_time = float(np.mean(perf_metrics['times']))
                max_time = float(np.max(perf_metrics['times']))
            if 'mcu_rom' in perf_metrics:
                mcu_rom = perf_metrics['mcu_rom']
            if 'mcu_ram' in perf_metrics:
                mcu_ram = perf_metrics['mcu_ram']

        # Calculate percentiles for error distribution
        error_percentiles = {
            '50th': float(np.percentile(absolute_errors, 50)),
            '90th': float(np.percentile(absolute_errors, 90)),
            '95th': float(np.percentile(absolute_errors, 95)),
            '99th': float(np.percentile(absolute_errors, 99))
        }
        
        return {
            'overall_metrics': {
                'avg_relative_error': float(avg_rel_error),
                'max_relative_error': float(max_rel_error),
                'mae': float(mae),
                'max_absolute_error': float(max_abs_error),
                'mse': float(mse),
                'rmse': float(rmse),
                'avg_time': avg_time,
                'max_time': max_time,
                'mcu_rom': mcu_rom,
                'mcu_ram': mcu_ram,
                'num_samples': len(onnx_results),
                'num_elements': len(absolute_errors)
            },
            'error_distribution': error_percentiles,
            'sample_errors': sample_errors[:5]  # First 5 samples for detailed view
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download_example/<filename>')
def download_example(filename):
    """Download example PyTorch scripts"""
    examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
    file_path = os.path.join(examples_dir, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "Example file not found", 404

@app.route('/task_status/<task_id>')
def task_status(task_id):
    """Get the current status and logs of a task"""
    with tasks_lock:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = tasks[task_id]
        return jsonify({
            'status': task['status'],
            'logs': task['logs'],
            'result': task.get('result'),
            'error': task.get('error')
        })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        # Generate unique ID for this conversion
        conversion_id = str(uuid.uuid4())
        task_id = conversion_id # Use same ID for task
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{conversion_id}_{filename}")
        file.save(upload_path)
        
        # Initialize task
        with tasks_lock:
            tasks[task_id] = {
                'status': 'in_progress',
                'logs': [],
                'result': None,
                'error': None,
                'start_time': time.time()
            }
        
        # Start background conversion
        thread = threading.Thread(
            target=background_conversion,
            args=(task_id, upload_path, filename, conversion_id)
        )
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'conversion_id': conversion_id,
            'status': 'started'
        })
    
    return jsonify({'error': 'Invalid file type. Please upload an ONNX file.'})

@app.route('/download/<conversion_id>')
def download_file(conversion_id):
    c_file_path = os.path.join(app.config['GENERATED_FOLDER'], f"{conversion_id}_generated.c")
    
    if os.path.exists(c_file_path):
        return send_file(c_file_path, as_attachment=True, 
                        download_name=f"generated_{conversion_id}.c")
    
    return jsonify({'error': 'File not found'})

@app.route('/download/<conversion_id>/main')
def download_main_file(conversion_id):
    """下载main.c文件"""
    main_file_path = os.path.join(app.config['GENERATED_FOLDER'], f"{conversion_id}_main.c")
    
    if os.path.exists(main_file_path):
        return send_file(main_file_path, as_attachment=True, 
                        download_name=f"main_{conversion_id}.c")
    
    return jsonify({'error': 'Main file not found'})

@app.route('/download/<conversion_id>/executable')
def download_executable(conversion_id):
    """下载可执行文件"""
    executable_path = os.path.join(app.config['GENERATED_FOLDER'], f"{conversion_id}_model")
    
    if os.path.exists(executable_path):
        return send_file(executable_path, as_attachment=True, 
                        download_name=f"model_{conversion_id}")
    
    return jsonify({'error': 'Executable file not found'})

@app.route('/report/<conversion_id>')
def get_report(conversion_id):
    """Get detailed conversion and validation report"""
    result_path = os.path.join(app.config['GENERATED_FOLDER'], f"{conversion_id}_result.json")
    
    if os.path.exists(result_path):
        try:
            with open(result_path, 'r') as f:
                report = json.load(f)
            return render_template('report.html', report=report)
        except Exception as e:
            # If there's an error loading, still redirect or show error
            return redirect(url_for('index'))
    
    # Redirect to home if report not found
    return redirect(url_for('index'))

@app.route('/api/report/<conversion_id>')
def get_report_json(conversion_id):
    """Get detailed conversion and validation report as JSON"""
    result_path = os.path.join(app.config['GENERATED_FOLDER'], f"{conversion_id}_result.json")
    
    if os.path.exists(result_path):
        try:
            with open(result_path, 'r') as f:
                report = json.load(f)
            return jsonify(report)
        except Exception as e:
            return jsonify({'error': f'Failed to load report: {str(e)}'})
    
    return jsonify({'error': 'Report not found'})

@app.route('/api/cleanup', methods=['POST'])
def manual_cleanup():
    """手动触发文件清理"""
    try:
        cleanup_old_files()
        return jsonify({'success': True, 'message': 'Cleanup completed successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Cleanup failed: {str(e)}'})

@app.route('/api/status')
def get_status():
    """获取服务器状态信息"""
    try:
        upload_files = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]) if os.path.exists(app.config['UPLOAD_FOLDER']) else 0
        generated_files = len([f for f in os.listdir(app.config['GENERATED_FOLDER']) if os.path.isfile(os.path.join(app.config['GENERATED_FOLDER'], f))]) if os.path.exists(app.config['GENERATED_FOLDER']) else 0
        
        return jsonify({
            'status': 'running',
            'upload_files_count': upload_files,
            'generated_files_count': generated_files,
            'cleanup_interval': '5 minutes',
            'file_retention': '10 minutes'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get status: {str(e)}'})

def create_main_c(output_path, conversion_id, model_path):
    """Create main.c file for testing the generated model"""
    
    # Get model input/output shapes from ONNX
    try:
        model = onnx.load(model_path)
        inputs_info = []
        outputs_info = []
        
        # Parse input tensors
        for input_tensor in model.graph.input:
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)  # Default for dynamic shapes
            inputs_info.append({
                'name': input_tensor.name,
                'shape': shape,
                'size': np.prod(shape)
            })
        
        # Parse output tensors
        for output_tensor in model.graph.output:
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)  # Default for dynamic shapes
            outputs_info.append({
                'name': output_tensor.name,
                'shape': shape,
                'size': np.prod(shape)
            })
            
    except Exception as e:
        print(f"Error parsing ONNX model: {e}")
        # Fallback to simple arrays
        inputs_info = [{'name': 'input', 'shape': [1, 784], 'size': 784}]
        outputs_info = [{'name': 'output', 'shape': [1, 10], 'size': 10}]
    
    # Generate input declarations
    input_declarations = []
    input_reads = []
    input_params = []
    
    for i, inp in enumerate(inputs_info):
        shape_str = ''.join(f'[{dim}]' for dim in inp['shape'])
        input_declarations.append(f"    float input_{i}{shape_str};")
        input_reads.append(f"    // Read input {i} data\n    for (int j = 0; j < {inp['size']}; j++) {{\n        if (fscanf(input_file, \"%f\", &value) == 1) {{\n            ((float*)input_{i})[j] = value;\n        }}\n    }}")
        input_params.append(f"input_{i}")
    
    # Generate output declarations
    output_declarations = []
    output_prints = []
    output_params = []
    
    for i, out in enumerate(outputs_info):
        shape_str = ''.join(f'[{dim}]' for dim in out['shape'])
        output_declarations.append(f"    float output_{i}{shape_str};")
        output_prints.append(f"    // Print output {i}\n    for (int j = 0; j < {out['size']}; j++) {{\n        printf(\"%.15g\\n\", ((float*)output_{i})[j]);\n    }}")
        output_params.append(f"output_{i}")
    
    input_decl_str = '\n'.join(input_declarations)
    input_read_str = '\n\n'.join(input_reads)
    output_decl_str = '\n'.join(output_declarations)
    output_print_str = '\n\n'.join(output_prints)
    
    input_params_str = ', '.join(input_params)
    output_params_str = ', '.join(output_params)
    
    main_c_content = f'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __linux__
#include <sys/resource.h>
#endif

// Forward declaration of the generated model entry function
extern void entry({', '.join([f'const float {inp["name"]}' + ''.join(f'[{dim}]' for dim in inp["shape"]) for inp in inputs_info])}, {', '.join([f'float {out["name"]}' + ''.join(f'[{dim}]' for dim in out["shape"]) for out in outputs_info])});

long get_max_rss() {{
#ifdef __linux__
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // in kilobytes
#else
    return 0;
#endif
}}

int main(int argc, char* argv[]) {{
    if (argc != 2) {{
        fprintf(stderr, "Usage: %s <input_file>\\n", argv[0]);
        return 1;
    }}
    
    FILE* input_file = fopen(argv[1], "r");
    if (!input_file) {{
        fprintf(stderr, "Error: Cannot open input file %s\\n", argv[1]);
        return 1;
    }}
    
    float value;
    
    // Declare input tensors
{input_decl_str}
    
    // Declare output tensors
{output_decl_str}
    
    // Read input data from file
{input_read_str}
    
    fclose(input_file);
    
    // Measure inference time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Call the entry function
    entry({input_params_str}, {output_params_str});
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_used = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Print results
{output_print_str}
    
    // Print timing and memory info to stderr
    fprintf(stderr, "\\n[METRICS] time: %.6f s\\n", time_used);
    fprintf(stderr, "[METRICS] memory: %ld KB\\n", get_max_rss());
    
    return 0;
}}'''
    
    with open(output_path, 'w') as f:
        f.write(main_c_content)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)