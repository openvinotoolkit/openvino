// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset13.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace ov {
namespace test {
using InputShapeAndTransposeOrder = std::pair<std::vector<InputShape>, std::vector<size_t>>;
using ConcatMultiQuerySDPParams = std::tuple<ElementType,
                                             InputShapeAndTransposeOrder,
                                             bool, // force kvcache int8
                                             bool  // has ShapeOf
                                             >;
// Subgraph:
/*                              Parameter
 *                                  |
 *       Parameter    ReadValue     |           ReadValue  Parameter
 *           \           /          |               \          /
 *            \       Gather        |             Gather      /
 *             \       /            |                 \      /
 *               Concat         Transpose              Concat
 *                / \               |                 /     \
 *               /   \              |                /       \
 *              /   MultiQuery      |           MultiQuery    \
 *             /       \            |              /           \
 *            /     Transpose       |          Transpose        \
 *           /           \          |            /               \
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

class ConcatMultiQuerySDPTest : public testing::WithParamInterface<ConcatMultiQuerySDPParams>,
                                virtual public ov::test::SubgraphBaseTest,
                                public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatMultiQuerySDPParams>& obj) {
        ElementType qkvType;
        InputShapeAndTransposeOrder inputShapeAndOrders;
        bool forceKVU8;
        bool hasShapeOf;
        std::tie(qkvType, inputShapeAndOrders, forceKVU8, hasShapeOf) = obj.param;
        ElementType kvCacheType = forceKVU8 ? ov::element::Type_t::u8 : qkvType;
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
        result << "qDataType=" << qkvType << "_";
        result << "kvDataType=" << kvCacheType << "_";
        result << "HasShapeOf=" << hasShapeOf << "_";
        result << "TransposeOrder=";
        result << "(";
        for (const auto& itr : transposeOrder) {
            result << itr << ",";
        }
        result << ")";

        return result.str();
    }

    void SetUp() override {
        InputShapeAndTransposeOrder inputShapeAndOrders;
        bool forceKVU8;
        bool hasShapeOf;
        ElementType qkvType;
        std::tie(qkvType, inputShapeAndOrders, forceKVU8, hasShapeOf) = this->GetParam();
        std::vector<InputShape>& inputShapes = inputShapeAndOrders.first;
        std::vector<size_t>& transposeOrder = inputShapeAndOrders.second;
        targetDevice = ov::test::utils::DEVICE_CPU;
        rel_threshold = 1e-2f;
        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        if (qkvType == ElementType::bf16 || qkvType == ElementType::f16) {
            configuration[ov::hint::inference_precision.name()] = ov::element::Type(qkvType).get_type_name();
            rel_threshold = 0.01f;
        }
        if (forceKVU8)
            configuration["KV_CACHE_PRECISION"] = "u8";
        init_input_shapes(inputShapes);
        ov::ParameterVector inputParams;
        // q,k,v
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(qkvType, inputDynamicShapes[0]));
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(qkvType, inputDynamicShapes[1]));
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(qkvType, inputDynamicShapes[1]));
        inputParams[0]->set_friendly_name("q");
        inputParams[1]->set_friendly_name("k");
        inputParams[2]->set_friendly_name("v");
        // pastkv init_cost
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(qkvType, inputDynamicShapes[2]));
        auto var_k = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{inputDynamicShapes[2], qkvType, "pastk"});
        auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_k);
        pastk->set_friendly_name("pastk_r");
        auto var_v = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{inputDynamicShapes[2], qkvType, "pastv"});
        auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_v);
        pastv->set_friendly_name("pastv_r");
        std::shared_ptr<Node> pastk_shapeof, pastv_shapeof;
        if (hasShapeOf) {
            pastk_shapeof = std::make_shared<ov::op::v0::ShapeOf>(pastk);
            pastv_shapeof = std::make_shared<ov::op::v0::ShapeOf>(pastv);
        }

        // pre SDPA transpose
        auto preOrder = ov::op::v0::Constant::create(ov::element::i32, {4}, transposeOrder);
        auto transposeQ = std::make_shared<ov::op::v1::Transpose>(inputParams[0], preOrder);

        auto concat_axis = transposeOrder[2];
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ElementType::i32, ov::PartialShape{-1});
        beam_idx->set_friendly_name("beam_idx");
        inputParams.push_back(beam_idx);
        auto gatherK = std::make_shared<ov::op::v8::Gather>(
            pastk,
            beam_idx,
            ov::op::v0::Constant::create(ElementType::i32, {1}, {transposeOrder[0]}));
        auto gatherV = std::make_shared<ov::op::v8::Gather>(
            pastv,
            beam_idx,
            ov::op::v0::Constant::create(ElementType::i32, {1}, {transposeOrder[0]}));
        auto concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherK, inputParams[1]}, concat_axis);
        auto concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherV, inputParams[2]}, concat_axis);

        auto unsquezeAxis = ov::op::v0::Constant::create(ov::element::i32, {}, {-2});
        auto unsqueezeK = std::make_shared<ov::op::v0::Unsqueeze>(concatK, unsquezeAxis);
        auto unsqueezeV = std::make_shared<ov::op::v0::Unsqueeze>(concatV, unsquezeAxis);

        auto targetShape = ov::op::v0::Constant::create(element::i32, {5}, {1, 1, 1, 4, 1});
        auto broadcastK = std::make_shared<ov::op::v3::Broadcast>(unsqueezeK, targetShape, op::BroadcastType::BIDIRECTIONAL);
        auto broadcastV = std::make_shared<ov::op::v3::Broadcast>(unsqueezeV, targetShape, op::BroadcastType::BIDIRECTIONAL);

        auto target4D = ov::op::v0::Constant::create(ov::element::i32, {4}, {0, 0, 8, 64});

        auto reshapeK = std::make_shared<ov::op::v1::Reshape>(broadcastK, target4D, true);
        auto reshapeV = std::make_shared<ov::op::v1::Reshape>(broadcastV, target4D, true);

        auto transposeK = std::make_shared<ov::op::v1::Transpose>(reshapeK, preOrder);
        auto transposeV = std::make_shared<ov::op::v1::Transpose>(reshapeV, preOrder);

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
            ov::op::v0::Constant::create(ov::element::i32, {4}, std::vector<size_t>{2, 0, 1, 3});  // BHLS -> LBHS
        auto transposeSDP = std::make_shared<ov::op::v1::Transpose>(sdp, postOrder);

        auto constReshape = ov::op::v0::Constant::create(ov::element::i32, {3}, reshapeOrder);
        auto reshapeSDP = std::make_shared<ov::op::v1::Reshape>(transposeSDP, constReshape, true);  // BLHS -> B,L,HxS

        auto add = std::make_shared<ov::op::v1::Add>(reshapeSDP, ov::op::v0::Constant::create(qkvType, {1}, {1.0f}));
        auto pastk_assign = std::make_shared<ov::op::v6::Assign>(concatK, var_k);
        auto pastv_assign = std::make_shared<ov::op::v6::Assign>(concatV, var_v);
        pastk_assign->set_friendly_name("pastk_w");
        pastv_assign->set_friendly_name("pastv_w");

        ov::OutputVector results{add};
        if (hasShapeOf) {
            results.push_back(pastk_shapeof);
            results.push_back(pastv_shapeof);
        }
        SinkVector sinks{pastk_assign, pastv_assign};
        function = std::make_shared<ov::Model>(results, sinks, inputParams, "ConcatTranposeSDP");
        targetDevice = ov::test::utils::DEVICE_CPU;

        functionRefs = function->clone();
        pass::Manager manager;
        // decompose ScaledDotProductAttention
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        std::vector<ov::Shape> shapes(4);
        shapes[0] = targetInputStaticShapes[0];
        shapes[1] = targetInputStaticShapes[1];
        shapes[2] = targetInputStaticShapes[1];
        shapes[3] = targetInputStaticShapes[2];
        SubgraphBaseTest::generate_inputs(shapes);
    }
    template <typename IT, typename T>
    void strided_iota(IT first, size_t n, T value, T stride) {
        for (size_t i = 0; i < n; i++) {
            *first++ = value;
            value += stride;
        }
    }
    void generate(int idx, const std::vector<ov::Shape>& targetInputStaticShapes) {
        inputs.clear();
        auto create_input = [this](std::shared_ptr<ov::op::v0::Parameter> param, ov::Shape shape, float val) {
            if (param->get_element_type() == element::i32) {
                ov::Tensor t{ov::element::i32, shape};
                auto size = shape[0];
                auto* p = static_cast<int*>(t.data());
                auto start = static_cast<int>(val);
                for (size_t i = 0; i < size; i++) {
                    p[i] = (start + i) % size;
                }
                inputs.insert({param, t});
            } else if (param->get_element_type() == element::f32) {
                ov::Tensor t{ov::element::f32, shape};
                strided_iota(static_cast<float*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            } else if (param->get_element_type() == element::f16) {
                ov::Tensor t{ov::element::f16, shape};
                strided_iota(static_cast<ov::float16*>(t.data()), t.get_size(), val, 0.0f);
                inputs.insert({param, t});
            } else {
                ov::Tensor t{ov::element::bf16, shape};
                strided_iota(static_cast<ov::bfloat16*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            }
        };
        // q, k, v, pastkv
        create_input(function->get_parameters()[0], targetInputStaticShapes[0], idx + 1.0f);
        create_input(function->get_parameters()[1], targetInputStaticShapes[1], idx + 2.0f);
        create_input(function->get_parameters()[2], targetInputStaticShapes[1], idx + 3.0f);
        create_input(function->get_parameters()[3], targetInputStaticShapes[2], idx + 4.0f);
        create_input(function->get_parameters()[4], ov::Shape{targetInputStaticShapes[0][1]}, idx + 0.0f);
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
        }
        auto states = inferRequest.query_state();
        // k, v may be in any order
        std::sort(states.begin(), states.end(), [] (VariableState& a, VariableState& b) {
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

TEST_P(ConcatMultiQuerySDPTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InputShapeAndTransposeOrder inputShapeAndOrders;
    bool forceKVU8;
    bool hasShapeOf;
    ElementType qkvType;
    std::tie(qkvType, inputShapeAndOrders, forceKVU8, hasShapeOf) = this->GetParam();
    auto actualOutputs = run_test(function);
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 1);
    CheckNumberOfNodesWithType(compiledModel, "Concatenation", 0);
    CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
    CheckNumberOfNodesWithType(compiledModel, "Transpose", 1);
    CheckNumberOfNodesWithType(compiledModel, "Gather", 0);
    // use f32 as reference
    if (qkvType == ElementType::f16) {
        configuration["INFERENCE_PRECISION_HINT"] = "f32";
    }
    auto expectedOutputs = run_test(functionRefs);
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 0);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {
const std::vector<InputShapeAndTransposeOrder> inputShapeAndReorders = {{
    {// inputShapes ChatGLM, greedy search
     {
         // L1, B, H, S small odd 1st token length < M_blk
         {{-1, 1, 8, 64}, {{5, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}}},
         {{-1, 1, 2, 64}, {{5, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}}},
         // L0, B, H, S
         {{-1, 1, 2, 64}, {{0, 1, 2, 64}, {6, 1, 2, 64}, {7, 1, 2, 64}, {8, 1, 2, 64}, {9, 1, 2, 64}}},
     },
     // transposeOrder
     {1, 2, 0, 3}},
    {// inputShapes ChatGLM, greedy search
     {
         // L1, B, H, S small even 1st token length < M_blk
         {{-1, 1, 8, 64}, {{16, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}}},
         {{-1, 1, 2, 64}, {{16, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}}},
         // L0, B, H, S
         {{-1, 1, 2, 64}, {{0, 1, 2, 64}, {17, 1, 2, 64}, {18, 1, 2, 64}, {19, 1, 2, 64}, {20, 1, 2, 64}}},
     },
     // transposeOrder
     {1, 2, 0, 3}},
    {// inputShapes ChatGLM, greedy search
     {
         // L1, B, H, S odd 1st token length > M_blk
         {{-1, 1, 8, 64}, {{33, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}}},
         {{-1, 1, 2, 64}, {{33, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}}},
         // L0, B, H, S
         {{-1, 1, 2, 64}, {{0, 1, 2, 64}, {34, 1, 2, 64}, {35, 1, 2, 64}, {36, 1, 2, 64}, {37, 1, 2, 64}}},
     },
     // transposeOrder
     {1, 2, 0, 3}},
    {// inputShapes ChatGLM, greedy search
     {
         // L1, B, H, S even 1st token length = M_blk
         {{-1, 1, 8, 64}, {{32, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}}},
         {{-1, 1, 2, 64}, {{32, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}}},
         // L0, B, H, S
         {{-1, 1, 2, 64}, {{0, 1, 2, 64}, {33, 1, 2, 64}, {34, 1, 2, 64}, {35, 1, 2, 64}, {36, 1, 2, 64}}},
     },
     // transposeOrder
     {1, 2, 0, 3}},
    {// inputShapes ChatGLM, even 1st token greedy search
     {
         // L1, B, H, S even 1st token length = 2 * M_blk
         {{-1, 1, 8, 64}, {{64, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}}},
         {{-1, 1, 2, 64}, {{64, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}}},
         // L0, B, H, S
         {{-1, 1, 2, 64}, {{0, 1, 2, 64}, {65, 1, 2, 64}, {66, 1, 2, 64}, {67, 1, 2, 64}, {68, 1, 2, 64}}},
     },
     // transposeOrder
     {1, 2, 0, 3}},
    {// inputShapes ChatGLM, even 1st token greedy search
     {
         // L1, B, H, S odd 1st token length > 2 * M_blk
         {{-1, 1, 8, 64}, {{65, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}}},
         {{-1, 1, 2, 64}, {{65, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}, {1, 1, 2, 64}}},
         // L0, B, H, S
         {{-1, 1, 2, 64}, {{0, 1, 2, 64}, {66, 1, 2, 64}, {67, 1, 2, 64}, {68, 1, 2, 64}, {69, 1, 2, 64}}},
     },
     // transposeOrder
     {1, 2, 0, 3}},
    {// beam search
     {
         // L1, B, H, S
         {{-1, -1, 8, 64}, {{10, 4, 8, 64}, {1, 4, 8, 64}, {1, 4, 8, 64}, {1, 4, 8, 64}, {1, 4, 8, 64}}},
         {{-1, -1, 2, 64}, {{10, 4, 2, 64}, {1, 4, 2, 64}, {1, 4, 2, 64}, {1, 4, 2, 64}, {1, 4, 2, 64}}},
         // L0, B, H, S
         {{-1, -1, 2, 64}, {{0, 4, 2, 64}, {10, 4, 2, 64}, {11, 4, 2, 64}, {12, 4, 2, 64}, {13, 4, 2, 64}}},
     },
     // transposeOrder
     {1, 2, 0, 3}},
}};

INSTANTIATE_TEST_SUITE_P(smoke_ConcatMultiQuerySDPTest,
                         ConcatMultiQuerySDPTest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::bf16, ElementType::f16),
                                            ::testing::ValuesIn(inputShapeAndReorders),
                                            ::testing::Values(true, false),
                                            ::testing::Values(true, false)),
                         ConcatMultiQuerySDPTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
