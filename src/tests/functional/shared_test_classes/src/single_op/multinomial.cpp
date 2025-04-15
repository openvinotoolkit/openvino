// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/multinomial.hpp"

using namespace ov::test;

namespace ov {
namespace test {
std::string MultinomialLayerTest::getTestCaseName(const testing::TestParamInfo<MultinomialTestParams>& obj) {
    std::string test_type;
    ov::Tensor probs;
    ov::Tensor num_samples;
    ov::test::ElementType convert_type;
    bool with_replacement;
    bool log_probs;
    std::pair<uint64_t, uint64_t> global_op_seed;
    std::string device_name;

    std::tie(test_type, probs, num_samples, convert_type, with_replacement, log_probs, global_op_seed, device_name) =
        obj.param;

    uint64_t global_seed = global_op_seed.first;
    uint64_t op_seed = global_op_seed.second;

    const char separator = '_';
    std::ostringstream result;
    result << test_type << separator;
    result << "probs_shape=" << probs.get_shape().to_string() << separator;
    if (num_samples.get_element_type() == ov::test::ElementType::i32) {
        result << "num_samples=" << static_cast<int*>(num_samples.data())[0] << separator;
    } else {  // i64
        result << "num_samples=" << static_cast<long*>(num_samples.data())[0] << separator;
    }
    result << "inType=" << probs.get_element_type() << separator;
    result << "convert_type=" << convert_type << separator;
    result << "replace=" << ov::test::utils::bool2str(with_replacement) << separator;
    result << "log=" << ov::test::utils::bool2str(log_probs) << separator;
    result << "seed_g=" << global_seed << separator;
    result << "seed_o=" << op_seed << separator;
    result << "device=" << device_name;

    return result.str();
}

void MultinomialLayerTest::SetUp() {
    MultinomialTestParams test_params;

    std::string test_type;
    ov::Tensor probs;
    ov::Tensor num_samples;
    ov::test::ElementType convert_type;
    bool with_replacement;
    bool log_probs;
    std::pair<uint64_t, uint64_t> global_op_seed;

    std::tie(test_type, probs, num_samples, convert_type, with_replacement, log_probs, global_op_seed, targetDevice) =
        GetParam();

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
    num_samples_param->set_friendly_name("num_samples");
    inputs.push_back(num_samples_param);
    params.push_back(num_samples_param);

    auto multinomial = std::make_shared<ov::op::v13::Multinomial>(inputs[0],
                                                                  inputs[1],
                                                                  convert_type,
                                                                  with_replacement,
                                                                  log_probs,
                                                                  global_seed,
                                                                  op_seed);

    ov::ResultVector results{std::make_shared<ov::opset10::Result>(multinomial)};
    function = std::make_shared<ov::Model>(results, params, "Multinomial");
}

void MultinomialLayerTest::generate_inputs(const std::vector<ov::Shape>& target_shapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();

    auto& probs = func_inputs[0];
    inputs.insert({probs.get_node_shared_ptr(), m_probs});
    auto& num_samples = func_inputs[1];
    inputs.insert({num_samples.get_node_shared_ptr(), m_num_samples});
}
}  // namespace test
}  // namespace ov
