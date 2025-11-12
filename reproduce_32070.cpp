#include "openvino/runtime/core.hpp"
#include "openvino/op/ops.hpp"

static ov::Output<ov::Node> addInput(ov::ParameterVector &parameters)
{
	auto node = std::make_shared<ov::op::v0::Parameter>(
		ov::element::Type_t::f32, ov::PartialShape{1, 1, 240, 320});
	parameters.push_back(node);
	return node;
}

static ov::Output<ov::Node> addConv(ov::Output<ov::Node> &prevNode,
	int numOutput, const std::vector<int32_t> &pad,
	const std::vector<int32_t> &kernelSize, int group,
	const std::vector<size_t> &stride,
	const std::vector<size_t> &shape0,
	const std::vector<size_t> &shape1)
{
	std::vector<ptrdiff_t> padBegin = {pad[0], pad[0]};
	std::vector<ptrdiff_t> padEnd = {pad[0], pad[0]};
	ov::Output<ov::Node> output;
	if (pad.size() == 2)
	{
		padBegin = {pad[0], pad[1]};
		padEnd = {pad[0], pad[1]};
	}

	std::vector<size_t> dilation = {1, 1};

	size_t size = 1;
	for (size_t w : shape0)
		size *= w;
	std::vector<float> data(size);

	if (group == 1)
	{
		auto weightsConstant = std::make_shared<ov::op::v0::Constant>(
			ov::element::Type_t::f32, ov::Shape(shape0), data);

		auto convNode = std::make_shared<ov::op::v1::Convolution>(
			prevNode, weightsConstant, ov::Strides(stride),
			ov::CoordinateDiff(padBegin), ov::CoordinateDiff(padEnd),
			ov::Strides(dilation));
		output = convNode->output(0);
	}
	else
	{
		std::vector<size_t> shape;
		shape.push_back(group);
		shape.push_back(shape0[0] / group);
		for (size_t i = 1; i < shape0.size(); i++)
		{
			shape.push_back(shape0[i]);
		}

		auto weightsConstant = std::make_shared<ov::op::v0::Constant>(
			ov::element::Type_t::f32, ov::Shape(shape), data);

		auto convNode = std::make_shared<ov::op::v1::GroupConvolution>(
			prevNode, weightsConstant, ov::Strides(stride),
			ov::CoordinateDiff(padBegin), ov::CoordinateDiff(padEnd),
			ov::Strides(dilation));
		output = convNode->output(0);
	}

	size = 1;
	for (size_t w : shape1)
		size *= w;
	data.resize(size);
	data[0] = 1;

	std::vector<size_t> biasShape = {1, shape1[0], 1, 1};
	auto biasConstant = std::make_shared<ov::op::v0::Constant>(
		ov::element::Type_t::f32, ov::Shape(biasShape), data);

	auto biasNode = std::make_shared<ov::op::v1::Add>(output, biasConstant->output(0));
	return biasNode->output(0);
}

static ov::Output<ov::Node> addPrelu(ov::Output<ov::Node> &prevNode, int params)
{
	std::vector<float> data(params);
	auto slopeConstant = std::make_shared<ov::op::v0::Constant>(
		ov::element::Type_t::f32, ov::Shape({(size_t)params}), data);
	auto preluNode = std::make_shared<ov::op::v0::PRelu>(prevNode, slopeConstant);
	return preluNode->output(0);
}

int main()
{
	ov::Core core;
	ov::ParameterVector parameters{};
	ov::ResultVector results;
	ov::Output<ov::Node> output;

	output = addInput(parameters);
	output = addConv (output, 24, {1}, {3, 3}, 1, {1, 1}, {24, 1, 3, 3}, {24});
	output = addPrelu(output, 24);
	output = addConv (output, 24, {0, 0}, {3, 3}, 24, {2, 2}, {24, 1, 3, 3}, {24});
	output = addPrelu(output, 24);
	output = addConv (output, 40, {0}, {1}, 1, {1, 1}, {40, 24, 1, 1}, {40});
	output = addPrelu(output, 40);
	output = addConv (output, 40, {1, 1}, {3, 3}, 40, {1, 1}, {40, 1, 3, 3}, {40});
	output = addPrelu(output, 40);

	results.push_back(std::make_shared<ov::op::v0::Result>(output));

	std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, parameters);

	ov::CompiledModel cmodel = core.compile_model(model, "CPU");
	auto req = cmodel.create_infer_request();
	
	std::vector<float> input_data(320*240);
	req.set_input_tensor({ov::element::f32, {1, 1, 240, 320}, input_data.data()});
	
	req.infer();
	
	return 0;
}
