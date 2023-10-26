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
    result << "conv_type=" << convert_type << separator;
    result << "repl=" << ov::test::utils::bool2str(with_replacement) << separator;
    result << "log_p=" << ov::test::utils::bool2str(log_probs) << separator;
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

    auto probs_param = std::make_shared<ov::op::v0::Parameter>(ov::test::ElementType::f32, probs_shape.first);
    auto num_samples_param = std::make_shared<ov::op::v0::Parameter>(ov::test::ElementType::i32, num_samples_shape.first);

    ov::ParameterVector params;
    std::vector<std::shared_ptr<ov::Node>> inputs;
    probs_param->set_friendly_name("probs");
    inputs.push_back(probs_param);
    params.push_back(probs_param);
    num_samples_param->set_friendly_name("num_samples");
    inputs.push_back(num_samples_param);
    params.push_back(num_samples_param);

    auto multinomial = std::make_shared<ov::op::v13::Multinomial>(params[0],
                                                                  params[1],
                                                                  convert_type,
                                                                  with_replacement,
                                                                  log_probs,
                                                                  global_seed,
                                                                  op_seed);

    ov::ResultVector results{std::make_shared<ov::opset10::Result>(multinomial)};
    function = std::make_shared<ov::Model>(results, params, "MultinomialCPU");
}

TEST_P(MultinomialLayerTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Multinomial");
}
}  // namespace CPULayerTestsDefinitions
