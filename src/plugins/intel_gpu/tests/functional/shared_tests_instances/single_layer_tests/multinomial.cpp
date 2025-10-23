// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/single_op/multinomial.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/op/multinomial.hpp"
#include "openvino/opsets/opset10_decl.hpp"
namespace ov {
namespace test {
class MultinomialLayerTestGPU : virtual public MultinomialLayerTest {
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override;
private:
    ov::Tensor m_probs;
    ov::Tensor m_num_samples;
};

void MultinomialLayerTestGPU::SetUp() {
    MultinomialTestParams test_params;

    const auto& [test_type, probs, num_samples, convert_type, with_replacement, log_probs, global_op_seed, _targetDevice] = GetParam();
    targetDevice = _targetDevice;

    m_probs = probs;
    m_num_samples = num_samples;

    uint64_t global_seed = global_op_seed.first;
    uint64_t op_seed = global_op_seed.second;

    InputShape probs_shape;
    InputShape num_samples_shape;
    const ov::Shape probs_tensor_shape = probs.get_shape();
    const ov::Shape num_samples_tensor_shape = num_samples.get_shape();
    if (test_type == "static") {
        probs_shape = {ov::PartialShape(probs_tensor_shape), {probs_tensor_shape}};
        num_samples_shape = {ov::PartialShape(num_samples_tensor_shape), {num_samples_tensor_shape}};
    } else {  // dynamic
        probs_shape = {ov::PartialShape::dynamic(ov::Rank(probs_tensor_shape.size())), {probs_tensor_shape}};
        num_samples_shape = {ov::PartialShape::dynamic(ov::Rank(num_samples_tensor_shape.size())),
                             {num_samples_tensor_shape}};
    }
    init_input_shapes({probs_shape, num_samples_shape});

    ov::ParameterVector params;
    std::vector<std::shared_ptr<ov::Node>> inputs;

    auto probs_param = std::make_shared<ov::op::v0::Parameter>(probs.get_element_type(), probs_shape.first);
    probs_param->set_friendly_name("probs");
    inputs.push_back(probs_param);
    params.push_back(probs_param);

    auto num_samples_param =
        std::make_shared<ov::op::v0::Parameter>(num_samples.get_element_type(), num_samples_shape.first);
    auto num_samples_const =
        std::make_shared<ov::op::v0::Constant>(num_samples.get_element_type(), m_num_samples.get_shape(), m_num_samples.data());
    num_samples_const->set_friendly_name("num_samples");
    inputs.push_back(num_samples_const);
    params.push_back(num_samples_param);

    auto multinomial = std::make_shared<ov::op::v13::Multinomial>(inputs[0],
                                                                  num_samples_const,
                                                                  convert_type,
                                                                  with_replacement,
                                                                  log_probs,
                                                                  global_seed,
                                                                  op_seed);

    ov::ResultVector results{std::make_shared<ov::opset10::Result>(multinomial)};
    function = std::make_shared<ov::Model>(results, params, "Multinomial");
}

void MultinomialLayerTestGPU::generate_inputs(const std::vector<ov::Shape>& target_shapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();

    auto& probs = func_inputs[0];
    inputs.insert({probs.get_node_shared_ptr(), m_probs});
    auto& num_samples = func_inputs[1];
    inputs.insert({num_samples.get_node_shared_ptr(), m_num_samples});
}

TEST_P(MultinomialLayerTestGPU, Inference) {
    run();
};

namespace {

std::vector<int64_t> num_samples_2_i64 = {2};
std::vector<int64_t> num_samples_4_i64 = {4};

const std::vector<ov::Tensor> inputTensors = {
                    ov::test::utils::create_and_fill_tensor(ov::element::f32, {1, 32}),
                    ov::test::utils::create_and_fill_tensor(ov::element::f32, {2, 28}),
                    ov::test::utils::create_and_fill_tensor(ov::element::f16, {1, 32}),
                    ov::test::utils::create_and_fill_tensor(ov::element::f16, {2, 28})};

const std::vector<ov::Tensor> numSamples = {
                    ov::Tensor(ov::element::i64, {1}, num_samples_2_i64.data()),
                    ov::Tensor(ov::element::i64, {1}, num_samples_4_i64.data())};

const std::vector<bool> withReplacement = {
    false,
    true
};

const std::vector<bool> logProbes = {
    false,
    true
};

const std::vector<std::string> static_dynamic = {
    "static"
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Multinomial,
    MultinomialLayerTestGPU,
    testing::Combine(testing::ValuesIn(static_dynamic),
                     testing::ValuesIn(inputTensors),
                     testing::ValuesIn(numSamples),
                     testing::Values(ov::element::i64),
                     testing::ValuesIn(withReplacement),
                     testing::ValuesIn(logProbes),
                     testing::Values(std::pair<uint64_t, uint64_t>{0, 2}),
                     testing::Values(ov::test::utils::DEVICE_GPU)),
                     MultinomialLayerTestGPU::getTestCaseName);
} // anonymous namespace
} // namespace test
} // namespace ov
