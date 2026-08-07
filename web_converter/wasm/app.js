/* WebAssembly frontend for onnx2c.
 *
 * Expects onnx2c.js and onnx2c.wasm (produced by the Emscripten build) to be
 * present in the same directory as this file.
 */

let onnx2cModule = null;
let currentOnnxBytes = null;
let currentFileName = null;
let lastGeneratedCode = null;
let conversionLogs = [];

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const convertBtn = document.getElementById('convertBtn');
const status = document.getElementById('status');
const resultCard = document.getElementById('resultCard');
const codeOutput = document.getElementById('codeOutput');
const downloadBtn = document.getElementById('downloadBtn');

function showStatus(message, type = 'info') {
    status.textContent = message;
    status.className = 'status ' + type;
    status.style.display = 'block';
}

function hideStatus() {
    status.style.display = 'none';
}

function buildArgs() {
    const args = [];

    const funcName = document.getElementById('funcName').value.trim();
    if (funcName && funcName !== 'entry') {
        args.push('-f', funcName);
    }

    const precision = document.getElementById('precision').value;
    if (precision && precision !== '20') {
        args.push('-P', precision);
    }

    if (document.getElementById('noGlobals').checked) {
        args.push('-n');
    }

    if (document.getElementById('onlyInit').checked) {
        args.push('-i');
    }

    const optimizations = document.getElementById('optimizations').value;
    if (optimizations) {
        args.push('-p', optimizations);
    }

    return args;
}

async function loadModule() {
    if (onnx2cModule) return onnx2cModule;

    showStatus('Loading onnx2c WASM module...', 'info');
    try {
        // The Emscripten MODULARIZE build exports createOnnx2cModule.
        const createModule = window.createOnnx2cModule || (await import('./onnx2c.js')).default;
        onnx2cModule = await createModule({
            print: (text) => console.log('[onnx2c]', text),
            printErr: (text) => {
                console.error('[onnx2c]', text);
                conversionLogs.push(text);
            }
        });
        hideStatus();
        return onnx2cModule;
    } catch (err) {
        showStatus('Failed to load WASM module. Make sure onnx2c.js and onnx2c.wasm are in this folder. Error: ' + err.message, 'error');
        throw err;
    }
}

async function convert() {
    if (!currentOnnxBytes) {
        showStatus('Please select an ONNX file first.', 'error');
        return;
    }

    convertBtn.disabled = true;
    resultCard.style.display = 'none';
    showStatus('Converting...', 'info');
    conversionLogs = [];

    try {
        const module = await loadModule();
        const args = buildArgs();

        const argVector = new module.StringVector();
        args.forEach(a => argVector.push_back(a));

        const result = module.convertOnnxBytes(currentOnnxBytes, argVector);
        lastGeneratedCode = result;

        codeOutput.textContent = result;
        resultCard.style.display = 'block';
        showStatus('Conversion complete.', 'success');
    } catch (err) {
        let message = err.message || String(err);
        const stderr = conversionLogs.join('\n').trim();
        if (stderr) {
            message += '\n\nDetails:\n' + stderr;
        }
        showStatus('Conversion failed: ' + message, 'error');
        console.error(err);
    } finally {
        convertBtn.disabled = false;
    }
}

function handleFile(file) {
    if (!file.name.endsWith('.onnx')) {
        showStatus('Please select a file ending with .onnx', 'error');
        return;
    }

    currentFileName = file.name.replace(/\.onnx$/i, '');
    fileName.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
        currentOnnxBytes = new Uint8Array(e.target.result);
        showStatus(`Loaded ${currentOnnxBytes.length.toLocaleString()} bytes. Ready to convert.`, 'info');
        convertBtn.disabled = false;
    };
    reader.onerror = () => {
        showStatus('Failed to read file.', 'error');
    };
    reader.readAsArrayBuffer(file);
}

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

convertBtn.addEventListener('click', convert);

downloadBtn.addEventListener('click', () => {
    if (!lastGeneratedCode) return;
    const blob = new Blob([lastGeneratedCode], { type: 'text/x-csrc' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (currentFileName || 'model') + '_inference.c';
    a.click();
    URL.revokeObjectURL(url);
});

// Pre-load the module in the background so conversion feels snappy.
loadModule().catch(() => {});
