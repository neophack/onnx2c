/* This file is part of onnx2c.
 */
#pragma once

#include <string>
#include <vector>

/* Convert an ONNX model (passed as raw bytes) into C source code.
 *
 * @param onnx_bytes  the raw ONNX file contents
 * @param args        command-line style arguments, e.g. {"-f", "entry", "-n",
 *                    "model.onnx"}. The last positional argument is treated as
 *                    the input file name but is not read; the model comes from
 *                    onnx_bytes.
 * @return            the generated C source code as a string
 *
 * Throws std::runtime_error or calls ERROR()/exit() on invalid input, matching
 * the native CLI behaviour.
 */
std::string convert_onnx_to_c(const std::string& onnx_bytes,
    const std::vector<std::string>& args = {});
