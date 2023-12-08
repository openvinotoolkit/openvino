// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/is_inf.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "ie_plugin_config.hpp"

using namespace ov::test::subgraph;

std::string IsInfLayerTest::getTestCaseName(const testing::TestParamInfo<IsInfParams>& obj) {
    std::vector<InputShape> inputShapes;
    ElementType dataPrc;
    bool detectNegative, detectPositive;
    std::string targetName;
    std::map<std::string, std::string> additionalConfig;
    std::tie(inputShapes, detectNegative, detectPositive, dataPrc, targetName, additionalConfig) = obj.param;
    std::ostringstream result;

    result << "IS=(";
    for (size_t i = 0lu; i < inputShapes.size(); i++) {
        result << ov::test::utils::partialShape2str({inputShapes[i].first}) << (i < inputShapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < inputShapes.size(); j++) {
            result << ov::test::utils::vec2str(inputShapes[j].second[i]) << (j < inputShapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << ")_detectNegative=" << (detectNegative ? "True" : "False") << "_";
    result << "detectPositive=" << (detectPositive ? "True" : "False") << "_";
    result << "dataPrc=" << dataPrc << "_";
    result << "trgDev=" << targetName;

    if (!additionalConfig.empty()) {
        result << "_PluginConf";
        for (auto &item : additionalConfig) {
            if (item.second == InferenceEngine::PluginConfigParams::YES)
                result << "_" << item.first << "=" << item.second;
        }
    }

    return result.str();
}

void IsInfLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ElementType dataPrc;
    bool detectNegative, detectPositive;
    std::string targetName;
    std::map<std::string, std::string> additionalConfig;
    std::tie(shapes, detectNegative, detectPositive, dataPrc, targetDevice, additionalConfig) = this->GetParam();

    init_input_shapes(shapes);
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    ov::ParameterVector parameters;
    for (auto&& shape : inputDynamicShapes) {
        parameters.push_back(std::make_shared<ov::op::v0::Parameter>(dataPrc, shape));
    }
    parameters[0]->set_friendly_name("Data");

    ov::op::v10::IsInf::Attributes attributes {detectNegative, detectPositive};
    auto isInf = std::make_shared<ov::op::v10::IsInf>(parameters[0], attributes);
    ov::ResultVector results;
    for (int i = 0; i < isInf->get_output_size(); i++) {
        results.push_back(std::make_shared<ov::op::v0::Result>(isInf->output(i)));
    }

    function = std::make_shared<ov::Model>(results, parameters, "IsInf");
}

namespace {

template <typename T>
void fill_tensor(ov::Tensor& tensor, int32_t range, T startFrom) {
    auto pointer = tensor.data<T>();
    testing::internal::Random random(1);
    for (size_t i = 0; i < range; i++) {
        if (i % 7 == 0) {
            pointer[i] = std::numeric_limits<T>::infinity();
        } else if (i % 7 == 1) {
            pointer[i] = std::numeric_limits<T>::quiet_NaN();
        } else if (i % 7 == 3) {
            pointer[i] = -std::numeric_limits<T>::infinity();
        } else if (i % 7 == 5) {
            pointer[i] = -std::numeric_limits<T>::quiet_NaN();
        } else {
            pointer[i] = startFrom + static_cast<T>(random.Generate(range));
        }
    }
}

}  // namespace

void IsInfLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    const auto& input = funcInputs[0];

    int32_t range = std::accumulate(targetInputStaticShapes[0].begin(), targetInputStaticShapes[0].end(), 1, std::multiplies<uint32_t>());
    float startFrom = -static_cast<float>(range) / 2.f;
    auto tensor = ov::Tensor{ input.get_element_type(), targetInputStaticShapes[0]};

    if (input.get_element_type() == ov::element::f16) {
        fill_tensor<ov::float16>(tensor, range, startFrom);
    } else {
        fill_tensor<float>(tensor, range, startFrom);
    }

    inputs.insert({input.get_node_shared_ptr(), tensor});
}
