const fs = require('fs');
const path = require('path');

// Mock browser globals for the web-only Emscripten module.
globalThis.document = { currentScript: { src: 'file:///' + __filename.replace(/\\/g, '/') } };
globalThis.fetch = () => { throw new Error('fetch should not be called'); };

const createOnnx2cModule = require('./onnx2c.js');

// Usage: node test_wasm.js <model.onnx> [extra onnx2c args...]
//   e.g. node test_wasm.js lstm_model.onnx -d sequence_length:5
const modelPath = process.argv[2] || 'test_add.onnx';
const extraArgs = process.argv.slice(3);

const onnxBytes = fs.readFileSync(modelPath);
const uint8Array = new Uint8Array(onnxBytes);

// Emscripten uses globalThis.fetch, not Module.fetch.
globalThis.fetch = (url) => {
    const data = fs.readFileSync(url);
    return Promise.resolve({
        ok: true,
        arrayBuffer: () => Promise.resolve(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))
    });
};

(async () => {
    try {
        const module = await createOnnx2cModule({
            print: (text) => console.log('[onnx2c]', text),
            printErr: (text) => console.error('[onnx2c]', text),
            locateFile: (filename) => path.join(__dirname, filename)
        });

        const args = new module.StringVector();
        extraArgs.forEach((a) => args.push_back(a));
        const result = module.convertOnnxBytes(uint8Array, args);
        console.log('=== CONVERSION OUTPUT (first 500 chars) ===');
        console.log(result.substring(0, 500));
        console.log('...');
        console.log('Output length:', result.length);
    } catch (err) {
        console.error('FAILED:', err.message);
        process.exit(1);
    }
})();
