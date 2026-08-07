/* This file is part of onnx2c.
 *
 * WebAssembly entry point. Exposes the ONNX-to-C converter as a JavaScript
 * callable function via Emscripten Embind.
 */
#ifdef __EMSCRIPTEN__

#include <cstdint>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "convert.h"

using namespace emscripten;

/* Convert raw ONNX bytes (passed as a JS Uint8Array) to generated C source code.
 *
 * @param onnx_bytes  raw ONNX file contents as a Uint8Array
 * @param args        vector of command-line style option strings, e.g.
 *                    {"-f", "entry", "-n"}. The input file name is added
 *                    automatically.
 * @return            generated C source code, or an empty string on error.
 */
std::string convert_onnx_bytes(const emscripten::val& onnx_bytes,
    const std::vector<std::string>& args)
{
	// Read the JS Uint8Array directly into a C++ string without going through
	// JS string UTF-8 encoding, which would corrupt binary protobuf data.
	unsigned int length = onnx_bytes["length"].as<unsigned int>();
	std::string onnx_string(length, '\0');
	emscripten::val view{ emscripten::typed_memory_view(length,
	    reinterpret_cast<uint8_t*>(&onnx_string[0])) };
	view.call<void>("set", onnx_bytes);

	// Append a dummy input file name so that the CLI parser sees a valid
	// positional argument.
	std::vector<std::string> full_args = args;
	full_args.push_back("model.onnx");
	return convert_onnx_to_c(onnx_string, full_args);
}

EMSCRIPTEN_BINDINGS(onnx2c_module) {
	function("convertOnnxBytes", &convert_onnx_bytes);
	register_vector<std::string>("StringVector");
}

#endif // __EMSCRIPTEN__
