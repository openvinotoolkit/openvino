// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "internal_properties.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace ov::test;
using namespace CPUTestUtils;

namespace ov {
namespace test {
using InputShapeAndTransposeOrder = std::pair<std::vector<InputShape>, std::vector<size_t>>;
using ConcatSDPTransposeTestParams = std::tuple<ElementType,
                                                InputShapeAndTransposeOrder,
                                                bool,   // has ShapeOf
                                                bool,   // quantize by channel
                                                size_t  // group_size
                                                >;
// Subgraph:
/*                              Parameter
 *                                  |
 *       Parameter    ReadValue     |           ReadValue  Parameter
 *           \           /          |               \          /
 *         Gather       /           |             Gather      /
 *             \       /            |                 \      /
 *               Concat         Transpose              Concat
 *                / \               |                 /     \
 *               /   \              |                /       \
 *              /   Transpose       |          Transpose      \
 *             /       \            |            /             \
 *          Assign      ScaledDotProductAttention              Assign
 *                                  |
 *                               Tranpose
 *                                  |
 *                                Reshape
 *                                  |
 *                                 Add
 *                                  |
 *                                Result
 */

class ConcatSDPTransposeTestBase : public testing::WithParamInterface<ConcatSDPTransposeTestParams>,
                                   virtual public ov::test::SubgraphBaseTest,
                                   public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatSDPTransposeTestParams>& obj) {
        ElementType inType;
        InputShapeAndTransposeOrder inputShapeAndOrders;
        bool hasShapeof;
        bool quantKeyByChannel;
        size_t groupSize;
        std::tie(inType, inputShapeAndOrders, hasShapeof, quantKeyByChannel, groupSize) = obj.param;
        std::ostringstream result;
        std::vector<InputShape>& inputShapes = inputShapeAndOrders.first;
        std::vector<size_t>& transposeOrder = inputShapeAndOrders.second;
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
        result << "HasShapeOf=" << hasShapeof << "_";
        result << "quantKeyByChannel=" << quantKeyByChannel << "_";
        result << "groupSize=" << groupSize << "_";
        result << "TransposeOrder=";
        result << "(";
        for (const auto& itr : transposeOrder) {
            result << itr << ",";
        }
        result << ")";

        return result.str();
    }

    void SetUp() override {
        ElementType inType;
        InputShapeAndTransposeOrder inputShapeAndOrders;
        std::tie(inType, inputShapeAndOrders, hasShapeOf, quantKeyByChannel, keyGroupSize) = this->GetParam();
        std::vector<InputShape>& inputShapes = inputShapeAndOrders.first;
        transposeOrder = inputShapeAndOrders.second;
        targetDevice = ov::test::utils::DEVICE_CPU;
        rel_threshold = 1e-2f;
        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        configuration[ov::key_cache_group_size.name()] = keyGroupSize;
        configuration[ov::value_cache_group_size.name()] = keyGroupSize;
        configuration[ov::intel_cpu::key_cache_quant_mode.name()] =
            quantKeyByChannel ? ov::intel_cpu::CacheQuantMode::BY_CHANNEL : ov::intel_cpu::CacheQuantMode::BY_HIDDEN;
        // explictly set u8 kv_cache for QuantByChannel
        if (quantKeyByChannel) {
            configuration[ov::hint::kv_cache_precision.name()] = ov::element::u8;
            abs_threshold = 0.19f;
        }
        if (inType == ElementType::bf16) {
            configuration[ov::hint::inference_precision.name()] = ov::element::bf16;
            rel_threshold = 0.01f;
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
        std::shared_ptr<ov::Node> pastk_shapeof, pastv_shapeof;
        if (hasShapeOf) {
            pastk_shapeof = std::make_shared<ov::op::v0::ShapeOf>(pastk);
            pastv_shapeof = std::make_shared<ov::op::v0::ShapeOf>(pastv);
        }

        // pre SDPA transpose
        auto preOrder = ov::op::v0::Constant::create(ov::element::i32, {4}, transposeOrder);
        auto transposeQ = std::make_shared<ov::op::v1::Transpose>(inputParams[0], preOrder);
        std::shared_ptr<ov::Node> transposeQ_shapeof;
        if (hasShapeOf) {
            transposeQ_shapeof = std::make_shared<ov::op::v0::ShapeOf>(transposeQ);
        }

        auto concat_axis = transposeOrder[2];
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ElementType::i32, ov::PartialShape{-1});
        beam_idx->set_friendly_name("beam_idx");
        inputParams.push_back(beam_idx);
        auto gatherK = std::make_shared<ov::op::v8::Gather>(pastk,
                                                            beam_idx,
                                                            ov::op::v0::Constant::create(ElementType::i32, {1}, {0}));
        auto gatherV = std::make_shared<ov::op::v8::Gather>(pastv,
                                                            beam_idx,
                                                            ov::op::v0::Constant::create(ElementType::i32, {1}, {0}));
        auto concatK = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gatherK, inputParams[1]}, concat_axis);
        auto concatV = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gatherV, inputParams[2]}, concat_axis);
        auto transposeK = std::make_shared<ov::op::v1::Transpose>(concatK, preOrder);
        auto transposeV = std::make_shared<ov::op::v1::Transpose>(concatV, preOrder);

        auto sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(transposeQ, transposeK, transposeV, false);
        sdp->set_friendly_name("mha");

        // post SDPA transpose + reshape
        auto get_reshape_order = [](const ov::PartialShape& qkv_shape,
                                    const std::vector<size_t>& transposeOrder) -> std::vector<size_t> {
            assert(transposeOrder.size() == 4);
            auto H = qkv_shape[transposeOrder[1]].get_length();
            auto S = qkv_shape[transposeOrder[3]].get_length();
            return std::vector<size_t>{0, 0, static_cast<size_t>(H * S)};
        };
        const auto reshapeOrder = get_reshape_order(inputDynamicShapes[0], transposeOrder);

        auto postOrder =
            ov::op::v0::Constant::create(ov::element::i32, {4}, std::vector<size_t>{0, 2, 1, 3});  // BHLS -> BLHS
        auto transposeSDP = std::make_shared<ov::op::v1::Transpose>(sdp, postOrder);

        auto constReshape = ov::op::v0::Constant::create(ov::element::i32, {3}, reshapeOrder);
        auto reshapeSDP = std::make_shared<ov::op::v1::Reshape>(transposeSDP, constReshape, true);  // BLHS -> B,L,HxS

        auto add = std::make_shared<ov::op::v1::Add>(reshapeSDP, ov::op::v0::Constant::create(inType, {1}, {1.0f}));
        auto pastk_assign = std::make_shared<ov::op::v6::Assign>(concatK, var_k);
        auto pastv_assign = std::make_shared<ov::op::v6::Assign>(concatV, var_v);
        pastk_assign->set_friendly_name("pastk_w");
        pastv_assign->set_friendly_name("pastv_w");

        ov::OutputVector results{add};
        if (hasShapeOf) {
            results.push_back(pastk_shapeof);
            results.push_back(pastv_shapeof);
            results.push_back(transposeQ_shapeof);
        }
        ov::SinkVector sinks{pastk_assign, pastv_assign};
        function = std::make_shared<ov::Model>(results, sinks, inputParams, "ConcatTranposeSDP");
        targetDevice = ov::test::utils::DEVICE_CPU;

        functionRefs = function->clone();
        ov::pass::Manager manager;
        // decompose ScaledDotProductAttention
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        std::vector<ov::Shape> shapes(4);
        shapes[0] = targetInputStaticShapes[0];
        shapes[1] = targetInputStaticShapes[0];
        shapes[2] = targetInputStaticShapes[0];
        shapes[3] = targetInputStaticShapes[1];
        SubgraphBaseTest::generate_inputs(shapes);
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
        auto create_input = [this](std::shared_ptr<ov::op::v0::Parameter> param, ov::Shape shape, float val) {
            if (param->get_element_type() == ov::element::i32) {
                ov::Tensor t{ov::element::i32, shape};
                auto size = shape[0];
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
            } else if (param->get_element_type() == ov::element::f16) {
                ov::Tensor t{ov::element::f16, shape};
                strided_iota(static_cast<ov::float16*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            } else {
                ASSERT_TRUE(param->get_element_type() == ov::element::bf16);
                ov::Tensor t{ov::element::bf16, shape};
                strided_iota(static_cast<ov::bfloat16*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            }
        };
        // q, k, v, pastkv
        create_input(function->get_parameters()[0], targetInputStaticShapes[0], idx + 1.0f);
        create_input(function->get_parameters()[1], targetInputStaticShapes[0], idx + 2.0f);
        create_input(function->get_parameters()[2], targetInputStaticShapes[0], idx + 3.0f);
        create_input(function->get_parameters()[3], targetInputStaticShapes[1], idx + 4.0f);
        create_input(function->get_parameters()[4], ov::Shape{targetInputStaticShapes[0][0]}, idx + 0.0f);
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
    std::vector<size_t> transposeOrder;
    size_t keyGroupSize = 0;
    bool quantKeyByChannel = false;
    bool hasShapeOf;
};

class ConcatSDPTransposeTest : public ConcatSDPTransposeTestBase {
public:
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
        auto states = inferRequest.query_state();
        // k, v may be in any order
        std::sort(states.begin(), states.end(), [](VariableState& a, VariableState& b) {
            return a.get_name() > b.get_name();
        });
        for (std::string name : {"pastk", "pastv"}) {
            auto itr = std::find_if(states.begin(), states.end(), [&](const ov::VariableState& state) {
                return name == state.get_name();
            });
            OPENVINO_ASSERT(itr != states.end(), "Failed to find ", name, " state");
            const auto& state = *itr;
            auto state_tensor = state.get_state();
            ov::Tensor copy{state_tensor.get_element_type(), state_tensor.get_shape()};
            state_tensor.copy_to(copy);
            outputs.push_back(copy);
        }

        reset();

        return outputs;
    }
};

TEST_P(ConcatSDPTransposeTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto actualOutputs = run_test(function);
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 1);
    CheckNumberOfNodesWithType(compiledModel, "Concatenation", 0);
    CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
    CheckNumberOfNodesWithType(compiledModel, "Transpose", 1);
    // Transformation TSShapeOfForward will change:
    // ?->transpose->shapeof ==> ?-->shapeof->gather
    //                            |->transpose
    CheckNumberOfNodesWithType(compiledModel, "Gather", hasShapeOf ? 1 : 0);
    auto expectedOutputs = run_test(functionRefs);
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 0);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {
const std::vector<InputShapeAndTransposeOrder> inputShapeAndReorders = {
    {// greedy search
     {{
          // B, L1, H, S
          {{1, -1, 8, 64}, {{1, 10, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 20, 8, 64}, {1, 1, 8, 64}}},
          // B, L0, H, S
          {{1, -1, 8, 64}, {{1, 0, 8, 64}, {1, 10, 8, 64}, {1, 11, 8, 64}, {1, 12, 8, 64}, {1, 32, 8, 64}}},
      },
      // transposeOrder
      {0, 2, 1, 3}},
     // beam search
     {{
          // B, L1, H, S
          {{-1, -1, 8, 64}, {{4, 10, 8, 64}, {4, 1, 8, 64}, {4, 1, 8, 64}, {4, 1, 8, 64}, {4, 1, 8, 64}}},
          // B, L0, H, S
          {{-1, -1, 8, 64}, {{4, 0, 8, 64}, {4, 10, 8, 64}, {4, 11, 8, 64}, {4, 12, 8, 64}, {4, 13, 8, 64}}},
      },
      // transposeOrder
      {0, 2, 1, 3}}}};

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPTransposeTest,
                         ConcatSDPTransposeTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(inputShapeAndReorders),
                                            ::testing::Values(true, false),
                                            ::testing::Values(false),
                                            ::testing::Values(0)),
                         ConcatSDPTransposeTest::getTestCaseName);

const std::vector<InputShapeAndTransposeOrder> shapesWithGreedySearch = {
    {// greedy search
     {{
          // B, L1, H, S
          {{1, -1, 8, 64},
           {{1, 7, 8, 64}, {1, 1, 8, 64}, {1, 16, 8, 64}, {1, 1, 8, 64}, {1, 15, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}}},
          // B, L0, H, S
          {{1, -1, 8, 64},
           {{1, 0, 8, 64},
            {1, 7, 8, 64},
            {1, 8, 8, 64},
            {1, 24, 8, 64},
            {1, 25, 8, 64},
            {1, 41, 8, 64},
            {1, 42, 8, 64}}},
      },
      // transposeOrder
      {0, 2, 1, 3}}}};

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPTransposeByChannelTest,
                         ConcatSDPTransposeTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(shapesWithGreedySearch),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::Values(8)),
                         ConcatSDPTransposeTest::getTestCaseName);
}  //  namespace

class ConcatSDPTransposeTestSetState : public ConcatSDPTransposeTestBase {
public:
    void reduce_state() {
        auto states = inferRequest.query_state();
        for (auto&& state : states) {
            auto state_tensor = state.get_state();
            ov::Tensor copy{state_tensor.get_element_type(), state_tensor.get_shape()};
            state_tensor.copy_to(copy);
            auto new_shape = state_tensor.get_shape();
            ASSERT_GE(new_shape[transposeOrder[2]], 1);
            new_shape[transposeOrder[2]] -= 1;
            ov::Tensor new_state{state_tensor.get_element_type(), new_shape, copy.data()};
            state.set_state(new_state);
        }
    }
    void new_state(ov::element::Type& type, const ov::Shape& pastKVInitShape) {
        auto fill = [](ov::Tensor& t, float val) {
            auto shape = t.get_shape();
            if (t.get_element_type() == ov::element::f32) {
                strided_iota(static_cast<float*>(t.data()), t.get_size(), val, 0.1f);
            } else if (t.get_element_type() == ov::element::f16) {
                strided_iota(static_cast<ov::float16*>(t.data()), t.get_size(), val, 0.1f);
            } else {
                ASSERT_TRUE(t.get_element_type() == ov::element::bf16);
                strided_iota(static_cast<ov::bfloat16*>(t.data()), t.get_size(), val, 0.1f);
            }
        };
        float val = 0;
        auto states = inferRequest.query_state();
        for (auto&& state : states) {
            auto new_shape = pastKVInitShape;
            new_shape[transposeOrder[2]] = 3;
            ov::Tensor new_state{type, new_shape};
            fill(new_state, val);
            val += 0.13f;

            state.set_state(new_state);
        }
    }
    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model) {
        function = model;
        // on spr, all kvccache precision will be covered and all paths for get/set_state will be tested
        auto input_type = model->get_parameters()[0]->get_element_type();
        if (input_type == ov::element::f32) {
            configuration[ov::hint::kv_cache_precision.name()] = "f32";
        } else if (input_type == ov::element::bf16) {
            configuration[ov::hint::kv_cache_precision.name()] = "bf16";
        } else {
            configuration[ov::hint::kv_cache_precision.name()] = "u8";
        }
        prepare();
        std::vector<ov::Tensor> outputs;
        // case 1: initialization + pastkv reaches limitation, remove some state
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
            if (idx > 1) {
                reduce_state();
            }
        }

        // case 2: after reset, set_state at once
        auto pastKVType = inferRequest.query_state()[0].get_state().get_element_type();
        reset();
        new_state(pastKVType, targetStaticShapes[0][1]);
        idx = 0;
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

        return outputs;
    }
};

TEST_P(ConcatSDPTransposeTestSetState, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ElementType inType;
    InputShapeAndTransposeOrder inputShapeAndOrders;
    bool hasShapeOf;
    bool quantKeyByChannel;
    size_t groupSize;
    std::tie(inType, inputShapeAndOrders, hasShapeOf, quantKeyByChannel, groupSize) = this->GetParam();

    // skip bf16 test on avx512 platform
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    auto actualOutputs = run_test(function);
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 1);
    CheckNumberOfNodesWithType(compiledModel, "Concatenation", 0);
    CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
    CheckNumberOfNodesWithType(compiledModel, "Transpose", 1);
    CheckNumberOfNodesWithType(compiledModel, "Gather", 0);
    auto expectedOutputs = run_test(functionRefs);
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 0);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {
const std::vector<InputShapeAndTransposeOrder> inputShapeAndReordersSetState = {
    {// beam search
     {{
          // B, L1, H, S
          {{-1, -1, 8, 64}, {{4, 10, 8, 64}, {4, 1, 8, 64}, {4, 1, 8, 64}, {4, 1, 8, 64}}},
          // B, L0, H, S and init tensor
          {{-1, -1, 8, 64}, {{4, 2, 8, 64}, {4, 12, 8, 64}, {4, 13, 8, 64}, {4, 14, 8, 64}}},
      },
      // transposeOrder
      {0, 2, 1, 3}}}};

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPTransposeTestSetState,
                         ConcatSDPTransposeTestSetState,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::bf16, ElementType::f16),
                                            ::testing::ValuesIn(inputShapeAndReordersSetState),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0)),
                         ConcatSDPTransposeTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
