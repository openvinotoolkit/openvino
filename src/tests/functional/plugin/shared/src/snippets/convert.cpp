// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/convert.hpp"
#include "subgraph_converts.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string Convert::getTestCaseName(testing::TestParamInfo<ov::test::snippets::ConvertParams> obj) {
    std::vector<InputShape> inputShape;
    std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> types;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShape, types, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        result << "IS[" << i << "]=" << ov::test::utils::partialShape2str({inputShape[i].first}) << "_";
        result << "TS[" << i << "]=";
        for (const auto& shape : inputShape[i].second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
    }
    result << "IT=" << ov::test::utils::vec2str(types.first) << "_";
    result << "OT=" << ov::test::utils::vec2str(types.second) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void Convert::SetUp() {
    std::vector<InputShape> inputShape;
    std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> types;
    std::tie(inputShape, types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(inputShape);
    auto f = ov::test::snippets::ConvertFunction(inputDynamicShapes, types.first[0], types.second[0]);
    function = f.getOriginal();
    output_type = types.second.front();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

parameters Convert::generate_params_random() const {
    int32_t startFrom, range, resolution = 5;
    switch (output_type) {
        case ov::element::f32:
        case ov::element::i32:
        case ov::element::bf16:
        case ov::element::f16:
            startFrom = -10;
            range = 20;
            break;
        case ov::element::u8:
            startFrom = -10;
            range = 20;
            break;
        case ov::element::i8:
            startFrom = 117;
            range = 20;
            break;
        default:
            startFrom = 0;
            range = 10;
    }
    return {{ startFrom, range, resolution }};
}

void Convert::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    const auto params = generate_params_random();
    OPENVINO_ASSERT(params.size() == funcInputs.size(), "Incorrect count of parameters for random generation and inputs of function!");

    for (int i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;
        int32_t startFrom, range, resolution;
        std::tie(startFrom, range, resolution) = params[i];
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = startFrom;
        in_data.range = range;
        in_data.resolution = resolution;
        tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

void ConvertInput::SetUp() {
    std::vector<InputShape> inputShape;
    std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> types;
    std::tie(inputShape, types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(inputShape);
    auto f = ov::test::snippets::ConvertInputFunction(inputDynamicShapes, types.first[0], types.second[0]);
    function = f.getOriginal();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }

    if (types.first[0] == ov::element::f32 && types.second[0] == ov::element::bf16) {
        abs_threshold = 3e-2;
    }
}

parameters ConvertInput::generate_params_random() const {
    parameters params;
    const auto& funcInputs = function->inputs();
    for (int i = 0; i < funcInputs.size(); ++i) {
        int32_t startFrom, range, resolution = 1;
        switch (funcInputs[i].get_element_type()) {
            case ov::element::f32:
            case ov::element::bf16:
            case ov::element::f16:
                startFrom = -10;
                range = 20;
                resolution = 7;
                break;
            case ov::element::i32:
            case ov::element::i8:
                startFrom = -32;
                range = 64;
                break;
            case ov::element::u8:
                startFrom = 10;
                range = 20;
                break;
            default:
                startFrom = 0;
                range = 10;
        }
        params.push_back({ startFrom, range, resolution });
    }
    return params;
}

void ConvertOutput::SetUp() {
    std::vector<InputShape> inputShape;
    std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> types;
    std::tie(inputShape, types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(inputShape);

    auto f = ov::test::snippets::ConvertOutputFunction(inputDynamicShapes, types.first[0], types.second[0]);
    function = f.getOriginal();
    output_type = types.second.front();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }

    if (types.first[0] == ov::element::bf16 && types.second[0] == ov::element::f32) {
        abs_threshold = 4e-2;
    }
}

void ConvertStub::SetUp() {
    std::vector<InputShape> inputShape;
    std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> types;
    std::tie(inputShape, types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(inputShape);

    auto f = ov::test::snippets::ConvertStubFunction(inputDynamicShapes, types.first[0], types.second[0]);
    function = f.getOriginal();
    output_type = types.second.front();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

void ConvertPartialInputsAndResults::SetUp() {
    std::vector<InputShape> inputShape;
    std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> types;
    std::tie(inputShape, types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(inputShape);

    auto f = ov::test::snippets::ConvertPartialInputsAndResultsFunction(inputDynamicShapes, types.first, types.second);
    function = f.getOriginal();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

void ConvertManyOnInputs::SetUp() {
    std::vector<InputShape> inputShape;
    std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> types;
    std::tie(inputShape, types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(inputShape);

    auto f = ov::test::snippets::ConvertManyOnInputsFunction(inputDynamicShapes, types.first);
    function = f.getOriginal();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

void ConvertManyOnOutputs::SetUp() {
    std::vector<InputShape> inputShape;
    std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> types;
    std::tie(inputShape, types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(inputShape);

    auto f = ov::test::snippets::ConvertManyOnOutputsFunction(inputDynamicShapes, types.first);
    function = f.getOriginal();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

void ConvertManyOnInputOutput::SetUp() {
    std::vector<InputShape> inputShape;
    std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> types;
    std::tie(inputShape, types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(inputShape);

    auto f = ov::test::snippets::ConvertManyOnInputOutputFunction(inputDynamicShapes, types.first, types.second);
    function = f.getOriginal();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

TEST_P(Convert, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ConvertInput, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ConvertOutput, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ConvertStub, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ConvertPartialInputsAndResults, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ConvertManyOnInputs, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ConvertManyOnOutputs, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ConvertManyOnInputOutput, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
