/* This file is part of onnx2c.
 */
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "convert.h"
#include "options.h"

int main(int argc, const char* argv[])
{
	parse_cmdline_options(argc, argv);

	std::ifstream input(options.input_file, std::ios::binary);
	if (!input.good()) {
		std::cerr << "Error opening input file: \"" << options.input_file << "\"" << std::endl;
		exit(1); //	TODO: check out error numbers for a more accurate one
	}
	if (input.peek() == EOF) {
		std::cerr << "\"" << options.input_file << "\" is empty" << std::endl;
		exit(1);
	}

	std::string onnx_bytes((std::istreambuf_iterator<char>(input)),
	    std::istreambuf_iterator<char>());

	std::vector<std::string> args;
	for (int i = 1; i < argc; ++i)
		args.emplace_back(argv[i]);

	std::cout << convert_onnx_to_c(onnx_bytes, args);
}
