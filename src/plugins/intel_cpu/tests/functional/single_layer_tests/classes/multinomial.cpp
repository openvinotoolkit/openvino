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
    InputShape probs_shape;
    InputShape num_samples_shape;
    ov::test::ElementType convert_type;
    bool with_replacement;
    bool log_probs;
    uint64_t global_seed;
    uint64_t op_seed;
    CPUSpecificParams cpu_params;
    ov::AnyMap additional_config;

    std::tie(test_type,
             probs_shape,
             num_samples_shape,
             convert_type,
             with_replacement,
             log_probs,
             global_seed,
             op_seed,
             cpu_params,
             additional_config) = obj.param;

    std::ostringstream result;
    const char separator = '_';

    result << test_type << separator;
    result << "probs_shape=" << ov::test::utils::partialShape2str({probs_shape.first}) << separator;
    result << "num_shape=" << ov::test::utils::partialShape2str({num_samples_shape.first}) << separator;
    result << "convert_type=" << convert_type << separator;
    result << "replacement=" << ov::test::utils::bool2str(with_replacement) << separator;
    result << "log_probs=" << ov::test::utils::bool2str(log_probs) << separator;
    result << "seed_global=" << global_seed << separator;
    result << "seed_op=" << op_seed << separator;

    if (!additional_config.empty()) {
        result << "PluginConf={";
        for (const auto& conf_item : additional_config) {
            result << "_" << conf_item.first << "=";
            conf_item.second.print(result);
        }
        result << "}";
    }
    return result.str();
}

void MultinomialLayerTestCPU::SetUp() {
    MultinomialTestCPUParams test_params;

    std::string test_type;
    InputShape probs_shape;
    InputShape num_samples_shape;
    ov::test::ElementType convert_type;
    bool with_replacement;
    bool log_probs;
    uint64_t global_seed;
    uint64_t op_seed;
    CPUSpecificParams cpu_params;
    ov::AnyMap additional_config;

    std::tie(test_type,
             probs_shape,
             num_samples_shape,
             convert_type,
             with_replacement,
             log_probs,
             global_seed,
             op_seed,
             cpu_params,
             additional_config) = GetParam();

    targetDevice = ov::test::utils::DEVICE_CPU;
    updateSelectedType("ref_any", convert_type, additional_config);

    init_input_shapes({probs_shape, num_samples_shape});

    size_t probs_count = std::accumulate(probs_shape.second[0].begin(), probs_shape.second[0].end(), 1, std::multiplies<size_t>());
    std::vector<float> probs(probs_count, 0.0f);
    // std::cout << "C: " << probs_count << std::endl; 
    // for(size_t q = 0; q < 4; q++) {
    //     probs.at(4 * q - 1) = 1.0f;
    // }
    probs.at(3)  = 10.0f;
    probs.at(7)  = 10.0f;
    probs.at(11) = 10.0f;
    probs.at(15) = 10.0f;
    auto probs_const = ngraph::builder::makeConstant(ov::test::ElementType::f32, probs_shape.second[0], probs);
    auto num_samples_const = ngraph::builder::makeConstant(ov::test::ElementType::i32, num_samples_shape.second[0], std::vector<int>{1});

    auto probs_param = std::make_shared<ov::op::v0::Parameter>(ov::test::ElementType::f32, probs_shape.first);
    auto num_samples_param = std::make_shared<ov::op::v0::Parameter>(ov::test::ElementType::i32, num_samples_shape.first);

    ov::ParameterVector params;
    std::vector<std::shared_ptr<ov::Node>> inputs;
    // probs_param->set_friendly_name("probs");
    // inputs.push_back(probs_param);
    // params.push_back(probs_param);
    // num_samples_param->set_friendly_name("num_samples");
    // inputs.push_back(num_samples_param);
    // params.push_back(num_samples_param);

    probs_const->set_friendly_name("probs");
    num_samples_const->set_friendly_name("num_samples");
    inputs.push_back(probs_const);
    inputs.push_back(num_samples_const);

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

void MultinomialLayerTestCPU::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    for(size_t i = 0; i < expected.size(); i++) {
        for(size_t j = 0; j < expected[i].get_size(); j++) {
            std::cout << ((int*)expected[i].data())[j] << " ";
        } std::cout << "\n";
        for(size_t j = 0; j < actual[i].get_size(); j++) {
            std::cout << ((int*)actual[i].data())[j] << " ";
        } std::cout << "\n";
    }
    SubgraphBaseTest::compare(expected, actual);
}

TEST_P(MultinomialLayerTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Multinomial");
}
}  // namespace CPULayerTestsDefinitions
