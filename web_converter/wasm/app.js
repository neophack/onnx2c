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
const dimsCard = document.getElementById('dimsCard');
const dimList = document.getElementById('dimList');
const addDimBtn = document.getElementById('addDimBtn');

/* ------------------------------------------------------------------ *
 * Minimal protobuf reader for ONNX ModelProto.
 *
 * Only what we need: walk the ONNX wire format to collect the names of
 * every *dynamic* (named) input tensor dimension so the user can supply
 * concrete sizes via the onnx2c `-d name:value` flag.
 *
 * Relevant field numbers (see onnx.proto):
 *   ModelProto.graph            = 7  (message)
 *   GraphProto.input            = 11 (repeated message)
 *   ValueInfoProto.name         = 1  (string)
 *   ValueInfoProto.type         = 2  (message)
 *   TypeProto.tensor_type       = 1  (message)
 *   TypeProto.TensorType.elem_type = 1 (varint)
 *   TypeProto.TensorType.shape  = 2  (message)
 *   TensorShapeProto.dim        = 1  (repeated message)
 *   Dimension.dim_value         = 1  (varint, present for fixed dims)
 *   Dimension.dim_param         = 2  (string, present for dynamic dims)
 * ------------------------------------------------------------------ */

function pbReadVarint(bytes, pos) {
    let result = 0;
    let shift = 0;
    let b;
    do {
        b = bytes[pos++];
        result |= (b & 0x7f) << shift;
        shift += 7;
    } while (b & 0x80);
    return [result >>> 0, pos];
}

// Read a length-delimited field body and return [bodyView, nextPos].
function pbReadBytes(bytes, pos) {
    const [len, afterLen] = pbReadVarint(bytes, pos);
    const body = bytes.subarray(afterLen, afterLen + len);
    return [body, afterLen + len];
}

function pbDecodeUtf8(bytes) {
    try {
        return new TextDecoder('utf-8').decode(bytes);
    } catch (e) {
        return null;
    }
}

// Iterate over top-level fields in `bytes`, invoking cb(tag, wireType, reader)
// where reader.peekPos() returns the start of the field value so callers can
// descend into nested messages.
function pbWalk(bytes, cb) {
    let pos = 0;
    const len = bytes.length;
    while (pos < len) {
        const start = pos;
        const [tagWire, afterTag] = pbReadVarint(bytes, pos);
        pos = afterTag;
        const tag = tagWire >>> 3;
        const wireType = tagWire & 0x07;
        let body = null;
        if (wireType === 0) {
            // varint
            const [val, after] = pbReadVarint(bytes, pos);
            pos = after;
            body = val;
        } else if (wireType === 2) {
            // length-delimited
            const [b, after] = pbReadBytes(bytes, pos);
            body = b;
            pos = after;
        } else if (wireType === 5) {
            // 32-bit
            body = bytes.subarray(pos, pos + 4);
            pos += 4;
        } else if (wireType === 1) {
            // 64-bit
            body = bytes.subarray(pos, pos + 8);
            pos += 8;
        } else {
            // Unknown wire type: cannot safely continue.
            return;
        }
        if (cb(tag, wireType, body, start) === false) return;
    }
}

// Extract {dimName: inputNames[]} for every dynamic input dimension.
// Returns an array of { name, inputs: [inputNames] } preserving first-seen order.
function extractDynamicDims(uint8) {
    const found = new Map(); // dimName -> Set(inputName)

    function recordDim(dimBytes, inputName) {
        // dim_param (field 2, string) marks a dynamic dimension. dim_value
        // (field 1, varint) marks a fixed one; we only care about dynamic.
        let paramName = null;
        pbWalk(dimBytes, (tag, wireType, body) => {
            if (tag === 2 && wireType === 2) { // dim_param
                paramName = pbDecodeUtf8(body);
                return false;
            }
        });
        if (paramName) {
            if (!found.has(paramName)) found.set(paramName, new Set());
            found.get(paramName).add(inputName);
        }
    }

    function walkShape(shapeBytes, inputName) {
        pbWalk(shapeBytes, (tag, wireType, body) => {
            if (tag === 1 && wireType === 2) { // TensorShapeProto.dim
                recordDim(body, inputName);
            }
        });
    }

    function walkTensorType(ttBytes, inputName) {
        pbWalk(ttBytes, (tag, wireType, body) => {
            if (tag === 2 && wireType === 2) { // shape
                walkShape(body, inputName);
            }
        });
    }

    function walkValueType(typeBytes, inputName) {
        pbWalk(typeBytes, (tag, wireType, body) => {
            if (tag === 1 && wireType === 2) { // tensor_type
                walkTensorType(body, inputName);
            }
        });
    }

    function walkInput(inputBytes) {
        let inputName = '';
        pbWalk(inputBytes, (tag, wireType, body) => {
            if (tag === 1 && wireType === 2) { // name
                inputName = pbDecodeUtf8(body) || '';
            } else if (tag === 2 && wireType === 2) { // type
                walkValueType(body, inputName);
            }
        });
    }

    function walkGraph(graphBytes) {
        pbWalk(graphBytes, (tag, wireType, body) => {
            if (tag === 11 && wireType === 2) { // GraphProto.input
                walkInput(body);
            }
        });
    }

    try {
        pbWalk(uint8, (tag, wireType, body) => {
            if (tag === 7 && wireType === 2) { // ModelProto.graph
                walkGraph(body);
            }
        });
    } catch (e) {
        // Any parse error just means we couldn't auto-detect; the user can
        // still add dimensions manually.
        return [];
    }

    const result = [];
    for (const [name, inputs] of found) {
        result.push({ name, inputs: Array.from(inputs) });
    }
    return result;
}

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

    // Dynamic input dimensions: emit `-d name:value` for each defined row.
    for (const { name, value } of collectDimensions()) {
        args.push('-d', name + ':' + value);
    }

    return args;
}

// Read every row in the dimensions editor. Returns [{name, value}] for valid
// rows only (non-empty name, positive integer value). Invalid rows are skipped
// silently here so conversion is not blocked; callers may warn separately.
function collectDimensions() {
    const out = [];
    if (!dimList) return out;
    const rows = dimList.querySelectorAll('.dim-row');
    rows.forEach((row) => {
        const nameInput = row.querySelector('.dim-name');
        const valueInput = row.querySelector('.dim-value');
        if (!nameInput || !valueInput) return;
        const name = nameInput.value.trim();
        const raw = valueInput.value.trim();
        if (!name || !raw) return;
        if (!/^\d+$/.test(raw)) return;
        const value = parseInt(raw, 10);
        if (!Number.isFinite(value) || value <= 0) return;
        out.push({ name, value });
    });
    return out;
}

function renderDetectedDims(dims) {
    if (!dimList) return;
    dimList.innerHTML = '';

    if (!dims || dims.length === 0) {
        dimsCard.style.display = 'none';
        return;
    }

    dims.forEach((d) => {
        const hint = d.inputs && d.inputs.length
            ? ` (used by: ${d.inputs.join(', ')})`
            : '';
        addDimensionRow(d.name, 1, hint);
    });

    dimsCard.style.display = 'block';
}

function addDimensionRow(name = '', value = 1, hint = '') {
    const row = document.createElement('div');
    row.className = 'dim-row';

    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.className = 'dim-name';
    nameInput.value = name;
    nameInput.placeholder = 'dim name (e.g. sequence_length)';

    const valueInput = document.createElement('input');
    valueInput.type = 'number';
    valueInput.className = 'dim-value';
    valueInput.value = value;
    valueInput.min = 1;
    valueInput.step = 1;

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'dim-remove';
    removeBtn.textContent = '✕';
    removeBtn.title = 'Remove dimension';
    removeBtn.addEventListener('click', () => {
        row.remove();
        if (dimList.querySelectorAll('.dim-row').length === 0) {
            // Keep the card visible so users can still add rows manually.
        }
    });

    if (hint) {
        const hintEl = document.createElement('div');
        hintEl.className = 'dim-hint';
        hintEl.textContent = hint;
        row.appendChild(nameInput);
        row.appendChild(valueInput);
        row.appendChild(removeBtn);
        row.appendChild(hintEl);
    } else {
        row.appendChild(nameInput);
        row.appendChild(valueInput);
        row.appendChild(removeBtn);
    }

    dimList.appendChild(row);
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
        const detected = extractDynamicDims(currentOnnxBytes);
        renderDetectedDims(detected);
        const dimNote = detected.length
            ? ` Detected ${detected.length} dynamic dimension(s) — review values below.`
            : '';
        showStatus(`Loaded ${currentOnnxBytes.length.toLocaleString()} bytes. Ready to convert.${dimNote}`, 'info');
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

if (addDimBtn) {
    addDimBtn.addEventListener('click', () => {
        if (dimsCard) dimsCard.style.display = 'block';
        addDimensionRow('', 1);
    });
}

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
