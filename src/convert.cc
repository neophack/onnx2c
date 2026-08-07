/* This file is part of onnx2c.
 */
#include "convert.h"

#include <sstream>

#include "onnx.pb.h"

#include "graph.h"
#include "options.h"
#include "tensor.h"

std::string convert_onnx_to_c(const std::string& onnx_bytes,
    const std::vector<std::string>& args)
{
	onnx::ModelProto onnx_model;

	// Rebuild argv. The caller is responsible for providing any option flags and
	// a dummy/real input file name as the last positional argument.
	std::vector<const char*> argv;
	argv.reserve(args.size() + 1);
	argv.push_back("onnx2c");
	for (const auto& a : args)
		argv.push_back(a.c_str());

	parse_cmdline_options(static_cast<int>(argv.size()), argv.data());

	if (!onnx_model.ParseFromString(onnx_bytes)) {
		ERROR("Input is not a valid ONNX model");
	}

	std::ostringstream output;
	output.precision(options.output_precision);
	toC::Graph toCgraph(onnx_model);
	if (options.opt_fold_casts)
		toCgraph.fold_casts();
	if (options.opt_unionize)
		toCgraph.unionize_tensors();
	toCgraph.set_no_globals(options.no_globals);

	if (options.only_init) {
		toCgraph.print_initialization(output);
	}
	else {
		toCgraph.print_source(output, options.interface_func_name);
	}

	return output.str();
}
