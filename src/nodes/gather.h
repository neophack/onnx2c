/* This file is part of onnx2c.
 *
 * Gather node.
 */
namespace toC {

class Gather : public Node {
	public:
	Gather()
	{
		op_name = "Gather";
		axis = 0;
	}
	/* Node attributes */
	int axis;

	virtual void parseAttributes(onnx::NodeProto& node) override
	{
		for (const auto& a : node.attribute()) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if (a.name() == "axis")
				axis = parse_attribute_int(a);
			else
				ERROR("Unknown attribute for Gather: " + a.name());
		}
	}

	virtual void resolve(void) override
	{
		const Tensor* data = get_input_tensor(0);
		const Tensor* indices = get_input_tensor(1);
		name_input(0, "X");
		name_input(1, "indices");

		unsigned a = axis >= 0 ? axis : data->rank() + axis;
		assert(a < data->rank());

		Tensor* t = new Tensor;

		// output shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
		for (unsigned i = 0; i < a; i++)
			t->data_dim.push_back(data->data_dim[i]);
		for (unsigned i = 0; i < indices->rank(); i++)
			t->data_dim.push_back(indices->data_dim[i]);
		for (unsigned i = a + 1; i < data->rank(); i++)
			t->data_dim.push_back(data->data_dim[i]);

		t->data_type = data->data_type;

		if (data->data_buffer && indices->data_buffer) {
			t->isConst = true;
			t->data_buffer = malloc(t->data_num_elem() * t->data_elem_size());
			// TODO: implement constant folding for Gather.
			// For now, only handle the simple case used in shape calculations:
			// data is 1D (from Shape node), indices is a scalar.
			if (data->rank() == 1 && indices->rank() == 0) {
				int32_t idx = indices->get_data_element(0);
				if (idx < 0) idx += data->data_dim[0];
				if (t->data_type == onnx::TensorProto_DataType_INT64) {
					((int64_t*)t->data_buffer)[0] = ((int64_t*)data->data_buffer)[idx];
				} else if (t->data_type == onnx::TensorProto_DataType_INT32) {
					((int32_t*)t->data_buffer)[0] = ((int32_t*)data->data_buffer)[idx];
				} else {
					// Fallback or error
					free(t->data_buffer);
					t->data_buffer = nullptr;
					t->isConst = false;
				}
			} else {
				// Too complex for now, fall back to non-const
				free(t->data_buffer);
				t->data_buffer = nullptr;
				t->isConst = false;
			}
		}

		register_output(t, "Y");
	}

	virtual void print(std::ostream& dst) const override
	{
		const Tensor* data = get_input_tensor(0);
		const Tensor* indices = get_input_tensor(1);
		const Tensor* output = get_output_tensor(0);
		INDT_1 << "/* Gather" << std::endl;
		INDT_1 << "   axis = " << axis << std::endl;
		INDT_1 << " */" << std::endl;

		// The real axis number, counting from 0
		unsigned a = axis >= 0 ? axis : data->rank() + axis;

		std::string oidx = output->rank() == 0 ? "*Y" : "Y";
		for (unsigned r = 0; r < output->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (unsigned " << lv << "=0; ";
			dst << lv << "<" << output->data_dim[r] << "; ";
			dst << lv << "++)" << std::endl;

			oidx += "[" + lv + "]";
		}

		std::string didx = "X";
		for (unsigned r = 0; r < a; r++) {
			didx += "[i" + std::to_string(r) + "]";
		}
		didx += "[idx]";
		for (unsigned r = a + indices->rank(); r < output->rank(); r++) {
			didx += "[i" + std::to_string(r) + "]";
		}

		std::string iidx = indices->rank() == 0 ? "*indices" : "indices";
		for (unsigned r = 0; r < indices->rank(); r++) {
			iidx += "[i" + std::to_string(r + a) + "]";
		}

		INDT_1 << "{" << std::endl;
		INDT_2 << "int32_t idx = " << iidx << ";" << std::endl;
		INDT_2 << "idx = idx < 0 ? " << data->data_dim[a] << "+idx : idx;" << std::endl;
		INDT_2 << oidx << " = " << didx << ";" << std::endl;
		INDT_1 << "}" << std::endl;
	}
};
} // namespace toC
