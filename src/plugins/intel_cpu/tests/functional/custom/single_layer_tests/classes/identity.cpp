// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "identity.hpp"
#include "openvino/op/identity.hpp"
#include "utils/general_utils.h"
#include "common_test_utils/node_builders/constant.hpp"

using namespace CPUTestUtils;

namespace ov::test {

std::string IdentityLayerTestCPU::getTestCaseName(const testing::TestParamInfo<IdentityLayerTestCPUParamSet>& obj) {
    std::ostringstream result;

    result << "IS="          << std::get<0>(obj.param);
    result << "_Prc="        << std::get<1>(obj.param);
    result << "_ConstIn="    << utils::bool2str(std::get<2>(obj.param));
    result << CPUTestsBase::getTestCaseName(std::get<3>(obj.param));

    const auto& config = std::get<4>(obj.param);
    if (!config.empty()) {
        result << "_PluginConf={";
        for (const auto& conf_item : config) {
            result << "_" << conf_item.first << "=";
            conf_item.second.print(result);
        }
        result << "}";
    }

    return result.str();
}

void IdentityLayerTestCPU::SetUp() {
    targetDevice = utils::DEVICE_CPU;

    const auto& [output_shape, output_precision, const_input, cpu_params, additionalConfig] = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpu_params;

    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    if (ov::intel_cpu::contains_key_value(additionalConfig, {ov::hint::inference_precision.name(), ov::element::bf16})) {
        selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
    } else {
        selectedType = makeSelectedTypeStr(selectedType, output_precision);
    }

    std::vector<InputShape> in_shapes;

    if (!const_input) {
        in_shapes.push_back({{}, {{output_shape}}});
    } else {
        in_shapes.push_back({{output_shape}, {{output_shape}}});
    }
    init_input_shapes(in_shapes);

    const auto data = std::make_shared<ov::op::v0::Parameter>(output_precision, output_shape);
    data->set_friendly_name("data");

    const auto op = std::make_shared<ov::op::v16::Identity>(data);
    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(op)};

    function = std::make_shared<ov::Model>(results, ParameterVector{data}, "IdentityLayerTestCPU");
}

void IdentityLayerTestCPU::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();
    const auto& func_input = func_inputs[0];
    auto tensor = ov::test::utils::create_and_fill_tensor(func_input.get_element_type(), targetInputStaticShapes[0], utils::InputGenerateData());
    inputs.insert({func_input.get_node_shared_ptr(), tensor});
}

TEST_P(IdentityLayerTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Identity");
}

}  // namespace ov::test
