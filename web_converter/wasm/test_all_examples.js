/* Batch-convert every ONNX model under ../app/examples using the WASM build.
 *
 * Models with dynamic input dimensions get a default concrete size so onnx2c
 * can generate code. Run from this directory:
 *
 *   node test_all_examples.js
 *
 * Prints a summary table at the end.
 */
const fs = require('fs');
const path = require('path');

globalThis.document = { currentScript: { src: 'file:///' + __filename.replace(/\\/g, '/') } };
globalThis.fetch = () => { throw new Error('fetch should not be called'); };

const createOnnx2cModule = require('./onnx2c.js');
const examplesDir = path.join(__dirname, '..', 'app', 'examples');

// Default dynamic-dimension values to inject for known models. These mirror
// what the web UI auto-fills so the batch run exercises the same path.
const DEFAULT_DIMS = {
    'lstm_model.onnx': ['-d', 'sequence_length:5'],
    'lstm_with_hc.onnx': ['-d', 'batch_size:1', '-d', 'sequence_length:5'],
    'mlp_model.onnx': ['-d', 's77:1'],
};

globalThis.fetch = (url) => {
    const data = fs.readFileSync(url);
    return Promise.resolve({
        ok: true,
        arrayBuffer: () => Promise.resolve(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))
    });
};

function listOnnxFiles(dir) {
    return fs.readdirSync(dir)
        .filter((f) => f.endsWith('.onnx'))
        .map((f) => path.join(dir, f))
        .sort();
}

(async () => {
    let module;
    try {
        module = await createOnnx2cModule({
            print: () => {},
            printErr: () => {},
            locateFile: (filename) => path.join(__dirname, filename)
        });
    } catch (err) {
        console.error('Failed to load onnx2c WASM module:', err.message);
        process.exit(1);
    }

    const files = listOnnxFiles(examplesDir);
    const results = [];

    for (const file of files) {
        const name = path.basename(file);
        const bytes = new Uint8Array(fs.readFileSync(file));
        const dims = DEFAULT_DIMS[name] || [];
        const entry = { name, dims: dims.join(' ') || '-', ok: false, len: 0, error: '' };

        try {
            const args = new module.StringVector();
            dims.forEach((a) => args.push_back(a));
            const out = module.convertOnnxBytes(bytes, args);
            entry.ok = true;
            entry.len = out.length;
        } catch (err) {
            entry.error = (err.message || String(err)).split('\n')[0].slice(0, 80);
        }
        results.push(entry);
        console.log(`${entry.ok ? 'OK  ' : 'FAIL'} ${name.padEnd(26)} dims=${entry.dims.padEnd(24)} len=${entry.len}${entry.error ? '  err=' + entry.error : ''}`);
    }

    console.log('\n=== SUMMARY ===');
    const ok = results.filter((r) => r.ok).length;
    console.log(`${ok}/${results.length} models converted successfully.`);
    const failures = results.filter((r) => !r.ok);
    if (failures.length) {
        console.log('Failures:');
        failures.forEach((r) => console.log(`  - ${r.name}: ${r.error}`));
    }
})();
