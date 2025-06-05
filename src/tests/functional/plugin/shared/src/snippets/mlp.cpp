// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mlp.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_mlp.hpp"

namespace ov {
namespace test {
namespace snippets {

void MLP::SetUp() {
    std::pair<InputShape, std::vector<Shape>> shapes;
    MLPFunction::WeightFormat weightFormat;
    utils::ActivationTypes ActType;
    ov::element::Type prc;
    ov::AnyMap additional_config;

    std::tie(shapes, weightFormat, ActType, prc, ref_num_nodes, ref_num_subgraphs, targetDevice, additional_config) = this->GetParam();
    auto [inShape, weightsShapes] = shapes;
    init_input_shapes({inShape});

    const auto subgraph_model = ov::test::snippets::MLPFunction(inputDynamicShapes, weightsShapes, weightFormat, ActType);
    function = subgraph_model.getOriginal();

    configuration.insert(additional_config.begin(), additional_config.end());
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }

    inType = outType = prc;
    setInferenceType(prc);
}

std::string MLP::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MLPParams> obj) {
    auto [shapes, weightFormat, ActType, prc, num_nodes, num_subgraphs, target_device, additional_config] = obj.param;
    auto [inputShape, weightsShapes] = shapes;

    std::ostringstream result;
    result << "InputShape=" << inputShape << "_";
    result << "weightsShapes=" << ov::test::utils::vec2str(weightsShapes) << "_";
    result << "WeightFormat=" << weightFormat << "_";
    result << "ActType=" << ActType << "_";
    result << "Prc=" << prc << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << target_device << "_";

    if (!additional_config.empty()) {
        result << "_PluginConf";
        for (auto& item : additional_config) {
            result << "_" << item.first << "=" << item.second.as<std::string>();
        }
    }
    return result.str();
}

TEST_P(MLP, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
