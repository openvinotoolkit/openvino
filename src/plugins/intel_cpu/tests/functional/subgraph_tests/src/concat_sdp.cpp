// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset13.hpp>
#include <transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"

using namespace ov::test;
using namespace ngraph;
using namespace CPUTestUtils;
using namespace InferenceEngine;

namespace SubgraphTestsDefinitions {

using ConcatSDPTestParams = std::tuple<ElementType, std::vector<InputShape>>;
// Subgraph:
/*                            Parameter
 *                                |
 *       Parameter    ReadValue   |    ReadValue  Parameter
 *           \           /        |       \          /
 *            \         /         |        \        /
 *               Concat           |          Concat
 *                / \             |            / \
 *               /   \            |           /   \
 *              /     \           |          /     \
 *          Assign     ScaledDotProductAttention  Assign
 *                                |
 *                               Add
 *                                |
 *                              Result
 */

class ConcatSDPTest : public testing::WithParamInterface<ConcatSDPTestParams>, virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatSDPTestParams>& obj) {
        ElementType inType;
        std::vector<InputShape> inputShapes;
        std::tie(inType, inputShapes) = obj.param;
        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")_";
        }
        result << "Prc=" << inType;
        return result.str();
    }

    void SetUp() override {
        ElementType inType;
        std::vector<InputShape> inputShapes;
        std::tie(inType, inputShapes) = this->GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;
        if (inType == ElementType::bf16) {
            configuration.insert({"ENFORCE_BF16", "YES"});
        }
        init_input_shapes(inputShapes);
        ov::ParameterVector inputParams;
        // q,k,v
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
        inputParams[0]->set_friendly_name("q");
        inputParams[1]->set_friendly_name("k");
        inputParams[2]->set_friendly_name("v");
        // pastkv init_cost
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]));
        auto var_k = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{inputDynamicShapes[1], inType, "pastk"});
        auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_k);
        pastk->set_friendly_name("pastk_r");
        auto var_v = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{inputDynamicShapes[1], inType, "pastv"});
        auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_v);
        pastv->set_friendly_name("pastv_r");
        auto concatK = builder::makeConcat(OutputVector{pastk, inputParams[1]}, 2);
        auto concatV = builder::makeConcat(OutputVector{pastv, inputParams[2]}, 2);
        auto sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(inputParams[0], concatK, concatV, false);
        sdp->set_friendly_name("mha");
        auto add = std::make_shared<op::v1::Add>(sdp, op::v0::Constant::create(ngraph::element::f32, {1}, {1.0f}));
        auto pastk_assign = std::make_shared<op::v6::Assign>(concatK, var_k);
        auto pastv_assign = std::make_shared<op::v6::Assign>(concatV, var_v);
        pastk_assign->set_friendly_name("pastk_w");
        pastv_assign->set_friendly_name("pastv_w");

        ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
        SinkVector sinks{pastk_assign, pastv_assign};
        function = std::make_shared<Function>(results, sinks, inputParams, "ConcatSDP");
        targetDevice = ov::test::utils::DEVICE_CPU;

        functionRefs = function->clone();
        pass::Manager manager;
        // decompose ScaledDotProductAttention
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);
        rel_threshold = 1e-4f;
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        std::vector<ov::Shape> shapes(4);
        shapes[0] = targetInputStaticShapes[0];
        shapes[1] = targetInputStaticShapes[0];
        shapes[2] = targetInputStaticShapes[0];
        shapes[3] = targetInputStaticShapes[1];
        SubgraphBaseTest::generate_inputs(shapes);
    }
    void generate(int idx, const std::vector<ov::Shape>& targetInputStaticShapes) {
        std::vector<ov::Shape> shapes(4);
        shapes[0] = targetInputStaticShapes[0];
        shapes[1] = targetInputStaticShapes[0];
        shapes[2] = targetInputStaticShapes[0];
        shapes[3] = targetInputStaticShapes[1];
        inputs.clear();
        auto itTargetShape = shapes.begin();
        for (const auto& param : function->get_parameters()) {
            std::shared_ptr<ov::Node> inputNode = param;
            for (size_t i = 0; i < param->get_output_size(); i++) {
                for (const auto& node : param->get_output_target_inputs(i)) {
                    std::shared_ptr<ov::Node> nodePtr = node.get_node()->shared_from_this();
                    for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                        if (nodePtr->get_input_node_ptr(port)->shared_from_this() == inputNode->shared_from_this()) {
                            ov::Tensor t{ov::element::f32, *itTargetShape};
                            std::fill_n(static_cast<float*>(t.data()), t.get_size(), idx + 1.0f);
                            inputs.insert({param, t});
                            break;
                        }
                    }
                }
            }
            itTargetShape++;
        }
    }
    void prepare() {
        compile_model();
        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);
    }
    void reset() {
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }
        inferRequest = ov::InferRequest();
        compiledModel = ov::CompiledModel();
    }
    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model) {
        function = model;
        prepare();
        std::vector<ov::Tensor> outputs;
        int idx = 0;
        for (auto&& shapes : targetStaticShapes) {
            generate(idx++, shapes);
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            inferRequest.infer();
            auto outputTensor = inferRequest.get_output_tensor(0);
            ov::Tensor copy{outputTensor.get_element_type(), outputTensor.get_shape()};
            outputTensor.copy_to(copy);
            outputs.push_back(copy);
        }
        reset();

        return outputs;
    }
};

TEST_P(ConcatSDPTest, CompareWithRefs) {
    auto actualOutputs = run_test(function);
    auto expectedOutputs = run_test(functionRefs);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {
const std::vector<std::vector<InputShape>> inputShapes = {
    // dynamic batch
    {
        // B, H, L1, S
        {{1, 8, -1, 64}, {{1, 8, 10, 64}, {1, 8, 1, 64}, {1, 8, 1, 64}}},
        // B, H, L0, S
        {{1, 8, -1, 64}, {{1, 8, 0, 64}, {1, 8, 10, 64}, {1, 8, 11, 64}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPTest,
                         ConcatSDPTest,
                         ::testing::Combine(::testing::Values(ElementType::f32), ::testing::ValuesIn(inputShapes)),
                         ConcatSDPTest::getTestCaseName);

}  // namespace
}  // namespace SubgraphTestsDefinitions
