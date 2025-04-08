// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/is_inf.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
std::string IsInfLayerTest::getTestCaseName(const testing::TestParamInfo<IsInfParams>& obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    bool detect_negative, detect_positive;
    std::string target_name;
    ov::AnyMap additional_config;
    std::tie(shapes, detect_negative, detect_positive, model_type, target_name, additional_config) = obj.param;
    std::ostringstream result;

    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << ")_detect_negative=" << (detect_negative ? "True" : "False") << "_";
    result << "detect_positive=" << (detect_positive ? "True" : "False") << "_";
    result << "model_type=" << model_type << "_";
    result << "trgDev=" << target_name;

    for (auto const& config_item : additional_config) {
        result << "_config_item=" << config_item.first << "=" << config_item.second.as<std::string>();
    }
    return result.str();
}

void IsInfLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ElementType model_type;
    bool detect_negative, detect_positive;
    ov::AnyMap additional_config;
    std::tie(shapes, detect_negative, detect_positive, model_type, targetDevice, additional_config) = this->GetParam();

    init_input_shapes(shapes);
    configuration.insert(additional_config.begin(), additional_config.end());

    ov::ParameterVector parameters;
    for (auto&& shape : inputDynamicShapes) {
        parameters.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }
    parameters[0]->set_friendly_name("Data");

    ov::op::v10::IsInf::Attributes attributes {detect_negative, detect_positive};
    auto is_inf = std::make_shared<ov::op::v10::IsInf>(parameters[0], attributes);

    ov::ResultVector results;
    for (int i = 0; i < is_inf->get_output_size(); i++) {
        results.push_back(std::make_shared<ov::op::v0::Result>(is_inf->output(i)));
    }

    function = std::make_shared<ov::Model>(results, parameters, "IsInf");
}
} // namespace test
} // namespace ov
