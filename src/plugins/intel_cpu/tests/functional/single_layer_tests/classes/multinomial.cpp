// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial.hpp"

#include "ov_models/builders.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

std::string MultinomialLayerTestCPU::getTestCaseName(const testing::TestParamInfo<MultinomialTestCPUParams>& obj) {
    std::string test_type;
    ov::Tensor probs;
    ov::Tensor num_samples;
    ov::test::ElementType convert_type;
    bool with_replacement;
    bool log_probs;
    uint64_t global_seed;
    uint64_t op_seed;
    CPUSpecificParams cpu_params;
    ov::AnyMap additional_config;

    std::tie(test_type,
             probs,
             num_samples,
             convert_type,
             with_replacement,
             log_probs,
             global_seed,
             op_seed,
             cpu_params,
             additional_config) = obj.param;

    const char separator = '_';
    std::ostringstream result;
    result << test_type << separator;
    result << "probs_shape=" << probs.get_shape().to_string() << separator;
    if (num_samples.get_element_type() == ov::test::ElementType::i32) {
        result << "num_samples=" << static_cast<int*>(num_samples.data())[0] << separator;
    } else {  // i64
        result << "num_samples=" << static_cast<long*>(num_samples.data())[0] << separator;
    }
    result << "convert_type=" << convert_type << separator;
    result << "replace=" << ov::test::utils::bool2str(with_replacement) << separator;
    result << "log=" << ov::test::utils::bool2str(log_probs) << separator;
    result << "seed_g=" << global_seed << separator;
    result << "seed_o=" << op_seed << separator;

    if (!additional_config.empty()) {
        result << "PluginConf={";
        for (const auto& conf_item : additional_config) {
            result << "_" << conf_item.first << "=";
            conf_item.second.print(result);
        }
        result << "}";
    }
    return result.str();
};

void MultinomialLayerTestCPU::SetUp() {
    MultinomialTestCPUParams test_params;

    std::string test_type;
    ov::Tensor probs;
    ov::Tensor num_samples;
    ov::test::ElementType convert_type;
    bool with_replacement;
    bool log_probs;
    uint64_t global_seed;
    uint64_t op_seed;
    CPUSpecificParams cpu_params;
    ov::AnyMap additional_config;

    std::tie(test_type,
             probs,
             num_samples,
             convert_type,
             with_replacement,
             log_probs,
             global_seed,
             op_seed,
             cpu_params,
             additional_config) = GetParam();

    m_probs = probs;
    m_num_samples = num_samples;
    targetDevice = ov::test::utils::DEVICE_CPU;
    updateSelectedType("ref_any", convert_type, additional_config);

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
    function = std::make_shared<ov::Model>(results, params, "MultinomialCPU");
}

void MultinomialLayerTestCPU::generate_inputs(const std::vector<ov::Shape>& target_shapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();

    auto& probs = func_inputs[0];
    inputs.insert({probs.get_node_shared_ptr(), m_probs});
    auto& num_samples = func_inputs[1];
    inputs.insert({num_samples.get_node_shared_ptr(), m_num_samples});
};

void MultinomialLayerTestCPU::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    for (size_t i = 0; i < expected.size(); i++) {
        for (size_t j = 0; j < expected[i].get_size(); j++) {
            std::cout << ((int*)expected[i].data())[j] << " ";
        }
        std::cout << "\n";
        for (size_t j = 0; j < actual[i].get_size(); j++) {
            std::cout << ((int*)actual[i].data())[j] << " ";
        }
        std::cout << "\n";
    }
    SubgraphBaseTest::compare(expected, actual);
};

TEST_P(MultinomialLayerTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Multinomial");
}
}  // namespace CPULayerTestsDefinitions
