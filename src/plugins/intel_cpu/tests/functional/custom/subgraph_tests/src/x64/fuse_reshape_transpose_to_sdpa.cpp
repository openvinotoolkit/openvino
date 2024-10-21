// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "openvino/pass/manager.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace ov {
namespace test {

// Subgraph:
/*
 *             Parameter       Parameter
 *                  |              |
 *   Parameter   ReadValue     ReadValue
 *       |          |   \          |    \
 *    Reshape    Reshape Assign Reshape Assign
 *       |          |              |
 *   Transpose  Transpoe       Transpose
 *        \         |            /
 *      ScaledDotProductAttention
 *                  |
 *              Tranpose
 *                  |
 *               Reshape
 *                  |
 *                Result
 */

// <Input_shapes, [H,S]>
using InputShapeAndReshapeOrder = std::pair<std::vector<InputShape>, std::vector<int32_t>>;
using FuseSDPAReshapeTransposeTestParams = std::tuple<ElementType, InputShapeAndReshapeOrder>;
class FuseSDPAReshapeTransposeTest : virtual public ov::test::SubgraphBaseTest,
                                     public testing::WithParamInterface<FuseSDPAReshapeTransposeTestParams>,
                                     public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseSDPAReshapeTransposeTestParams>& obj) {
        ElementType inType;
        InputShapeAndReshapeOrder inputShapeAndOrders;
        std::tie(inType, inputShapeAndOrders) = obj.param;
        std::ostringstream result;
        std::vector<InputShape>& inputShapes = inputShapeAndOrders.first;
        auto& reshapeOrderHS = inputShapeAndOrders.second;
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
        result << "Prc=" << inType << "_";
        result << "ReshapeOrderHS=";
        result << "(";
        for (const auto& itr : reshapeOrderHS) {
            result << itr << ",";
        }
        result << ")";

        return result.str();
    }

    void SetUp() override {
        ElementType inType;
        InputShapeAndReshapeOrder inputShapeAndOrders;
        std::tie(inType, inputShapeAndOrders) = this->GetParam();
        std::vector<InputShape>& inputShapes = inputShapeAndOrders.first;
        auto& reshapeOrderHS = inputShapeAndOrders.second;
        targetDevice = ov::test::utils::DEVICE_CPU;
        rel_threshold = 1e-2f;
        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        if (inType == ElementType::bf16) {
            configuration[ov::hint::inference_precision.name()] = ov::element::bf16;
            rel_threshold = 0.01f;
        }
        init_input_shapes(inputShapes);

        // pre SDPA reshape->transpose
        ov::ParameterVector inputParams(3);
        ov::SinkVector sinkNodes;
        OutputVector transposes(3);
        for (size_t i = 0; i < 3u; i++) {
            inputParams[i] = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);

            auto reshape_axis =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 0, reshapeOrderHS[0], reshapeOrderHS[1]});

            std::shared_ptr<ov::Node> reshape_input_1 = inputParams[i];
            if (i > 0) {
                auto var = std::make_shared<ov::op::util::Variable>(
                    ov::op::util::VariableInfo{inputDynamicShapes[0], inType, "var_" + std::to_string(i)});
                auto readvalue = std::make_shared<ov::op::v6::ReadValue>(inputParams[i], var);
                auto assign = std::make_shared<ov::op::v6::Assign>(readvalue, var);
                sinkNodes.emplace_back(assign);
                reshape_input_1 = readvalue;
            }

            auto reshape = std::make_shared<ov::op::v1::Reshape>(reshape_input_1, reshape_axis, true);
            auto transposeOrder = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
            transposes[i] = std::make_shared<ov::op::v1::Transpose>(reshape, transposeOrder);
        }

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(transposes, false);
        sdpa->set_friendly_name("mha");

        // post SDPA transpose + reshape
        auto postOrder =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<size_t>{0, 2, 1, 3});  // BHLS -> BLHS
        auto transposeSDPA = std::make_shared<ov::op::v1::Transpose>(sdpa, postOrder);

        auto constReshape =
            ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 0, reshapeOrderHS[0] * reshapeOrderHS[1]});
        auto reshapeSDPA = std::make_shared<ov::op::v1::Reshape>(transposeSDPA, constReshape, true);  // BLHS -> B,L,HxS

        function = std::make_shared<ov::Model>(ov::OutputVector{reshapeSDPA},
                                               sinkNodes,
                                               inputParams,
                                               "FuseSDPAReshapeTranspose");
        targetDevice = ov::test::utils::DEVICE_CPU;
        functionRefs = function->clone();
        pass::Manager manager;
        // decompose ScaledDotProductAttention
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);
    }

    template <typename IT, typename T>
    static void strided_iota(IT first, size_t n, T value, T stride) {
        for (size_t i = 0; i < n; i++) {
            *first++ = value;
            value += stride;
        }
    }
    void generate(int idx, const std::vector<ov::Shape>& targetInputStaticShapes) {
        inputs.clear();
        auto create_input = [this] (std::shared_ptr<ov::op::v0::Parameter> param, ov::Shape shape, float val) {
            if (param->get_element_type() == ov::element::i32) {
                ov::Tensor t{ov::element::i32, shape};
                auto size = ov::shape_size<ov::Shape>(shape);
                auto* p = static_cast<int*>(t.data());
                auto start = static_cast<int>(val);
                for (size_t i = 0; i < size; i++) {
                    p[i] = (start + i) % size;
                }
                inputs.insert({param, t});
            } else if (param->get_element_type() == ov::element::f32) {
                ov::Tensor t{ov::element::f32, shape};
                strided_iota(static_cast<float*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            } else {
                ASSERT_TRUE(param->get_element_type() == ov::element::bf16);
                ov::Tensor t{ov::element::bf16, shape};
                strided_iota(static_cast<ov::bfloat16*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            }
        };
        // q, k, v
        create_input(function->get_parameters()[0], targetInputStaticShapes[0], idx + 1.0f);
        create_input(function->get_parameters()[1], targetInputStaticShapes[0], idx + 2.0f);
        create_input(function->get_parameters()[2], targetInputStaticShapes[0], idx + 3.0f);
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
            reset();
        }
        return outputs;
    }
};

TEST_P(FuseSDPAReshapeTransposeTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    bool reshape_transpose_fused = false;
    auto actualOutputs = run_test(function);
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 1);
    CheckNumberOfNodesWithType(compiledModel, "Reshape", 0);
    CheckNumberOfNodesWithType(compiledModel, "Transpose", 0);
    for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
        if (n->get_friendly_name() == "mha/fused_reshape_transpose") {
            reshape_transpose_fused = true;
        }
    }
    ASSERT_TRUE(reshape_transpose_fused);

    auto expectedOutputs = run_test(functionRefs);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {
const std::vector<InputShapeAndReshapeOrder> inputShapeAndReshapeOrders = {
    // <Input_shapes, [H,S]>
    {
        {{
             // Q,K,V:[B, L, H*S]
             {{-1, -1, 4 * 16}, {{1, 1, 4 * 16}, {1, 2, 4 * 16}, {2, 2, 4 * 16}}},
         },
         // reshapeOrderHS
         {4, 16}},
    }};

INSTANTIATE_TEST_SUITE_P(smoke_FuseSDPAReshapeTransposeTest,
                         FuseSDPAReshapeTransposeTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(inputShapeAndReshapeOrders)),
                         FuseSDPAReshapeTransposeTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
