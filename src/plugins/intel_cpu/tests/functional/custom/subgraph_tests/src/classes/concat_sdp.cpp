// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "concat_sdp.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

// Subgraph:
/*                            Parameter
 *                                |
 *       Parameter    ReadValue   |    ReadValue  Parameter
 *           \           /        |       \          /
 *         Gather       /               Gather      /
 *             \       /          |         \      /
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
std::string ConcatSDPTest::getTestCaseName(const testing::TestParamInfo<ConcatSDPTestParams>& obj) {
    ElementType inType;
    std::vector<InputShape> inputShapes;
    bool forceKVU8;
    bool hasShapeOf;
    bool isDiffKVHeadSize;
    std::tie(inType, inputShapes, forceKVU8, hasShapeOf, isDiffKVHeadSize) = obj.param;
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
    result << "Prc=" << inType << "_";
    result << "ForceKVU8=" << forceKVU8 << "_";
    result << "HasShapeOf=" << hasShapeOf << "_";
    result << "IsDiffKVHeadSize=" << isDiffKVHeadSize;
    return result.str();
}

void ConcatSDPTest::SetUp() {
    ElementType inType;
    std::vector<InputShape> inputShapes;
    std::tie(inType, inputShapes, m_forceKVU8, m_hasShapeOf, m_isDiffKVHeadSize) = this->GetParam();
    targetDevice = ov::test::utils::DEVICE_CPU;
    rel_threshold = 1e-2f;
    if (inType == ElementType::bf16 || inType == ElementType::f16) {
        configuration.insert({"INFERENCE_PRECISION_HINT", ov::element::Type(inType).get_type_name()});
        rel_threshold = 0.01f;
    }

    if (m_forceKVU8) {
        configuration["KV_CACHE_PRECISION"] = "u8";
    }
    init_input_shapes(inputShapes);
    ov::ParameterVector inputParams;
    // q,k,v
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
    auto v_ps = inputDynamicShapes[0];
    if (m_isDiffKVHeadSize) {
        v_ps[3] += m_diffKVHeadSize;
    }
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, v_ps));
    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");
    // pastkv init_cost
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]));
    auto var_k = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputDynamicShapes[1], inType, "pastk"});
    auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_k);
    pastk->set_friendly_name("pastk_r");
    // pastv init_cost
    auto v_init_ps = inputDynamicShapes[1];
    if (m_isDiffKVHeadSize) {
        v_init_ps[3] += m_diffKVHeadSize;
    }
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, v_init_ps));
    auto var_v = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{v_init_ps, inType, "pastv"});
    auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[4], var_v);
    pastv->set_friendly_name("pastv_r");
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ElementType::i32, ov::PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    inputParams.push_back(beam_idx);
    auto gatherK = std::make_shared<ov::op::v8::Gather>(pastk, beam_idx, op::v0::Constant::create(ElementType::i32, {}, {0}));
    auto gatherV = std::make_shared<ov::op::v8::Gather>(pastv, beam_idx, op::v0::Constant::create(ElementType::i32, {}, {0}));
    std::shared_ptr<Node> shapeof_k, shapeof_v;
    // test special case:
    // ReadValue->Gather->Concat->SDPA...
    //              |
    //            ShapeOf...
    // The transformation 'SimplifyGatherShapeOf' will move ShapeOf to be the child of ReadValue
    if (m_hasShapeOf) {
        shapeof_k = std::make_shared<ov::op::v0::ShapeOf>(gatherK);
        shapeof_v = std::make_shared<ov::op::v0::ShapeOf>(gatherV);
    }
    auto concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherK, inputParams[1]}, 2);
    auto concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherV, inputParams[2]}, 2);
    auto sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(inputParams[0], concatK, concatV, false);
    sdp->set_friendly_name("mha");
    auto add = std::make_shared<ov::op::v1::Add>(sdp, op::v0::Constant::create(inType, {1}, {1.0f}));
    auto pastk_assign = std::make_shared<op::v6::Assign>(concatK, var_k);
    auto pastv_assign = std::make_shared<op::v6::Assign>(concatV, var_v);
    pastk_assign->set_friendly_name("pastk_w");
    pastv_assign->set_friendly_name("pastv_w");

    ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    if (m_hasShapeOf) {
        results.push_back(std::make_shared<ov::op::v0::Result>(shapeof_k));
        results.push_back(std::make_shared<ov::op::v0::Result>(shapeof_v));
    }
    SinkVector sinks{pastk_assign, pastv_assign};
    function = std::make_shared<ov::Model>(results, sinks, inputParams, "ConcatSDP");
    targetDevice = ov::test::utils::DEVICE_CPU;
    functionRefs = function->clone();
    pass::Manager manager;
    // decompose ScaledDotProductAttention
    manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    manager.run_passes(functionRefs);
}

template<typename IT, typename T>
void strided_iota(IT first, size_t n, T value, T stride) {
    for (size_t i = 0; i < n; i++) {
        *first++ = value;
        value += stride;
    }
}

void ConcatSDPTest::generate(int idx, const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto create_input = [this] (std::shared_ptr<op::v0::Parameter> param, ov::Shape shape, float val) {
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
            strided_iota(static_cast<ov::float16 *>(t.data()), t.get_size(), val, 0.0f);
            inputs.insert({param, t});
        } else {
            ASSERT_TRUE(param->get_element_type() == element::bf16);
            ov::Tensor t{ov::element::bf16, shape};
            strided_iota(static_cast<ov::bfloat16*>(t.data()), t.get_size(), val, 0.1f);
            inputs.insert({param, t});
        }
    };
    // q, k, v, pastkv
    auto v_shape = targetInputStaticShapes[0];
    auto v_init_shape = targetInputStaticShapes[1];
    if (m_isDiffKVHeadSize) {
        v_shape[3] += m_diffKVHeadSize;
        v_init_shape[3] += m_diffKVHeadSize;
    }

    create_input(function->get_parameters()[0], targetInputStaticShapes[0], idx + 1.0f);
    create_input(function->get_parameters()[1], targetInputStaticShapes[0], idx + 2.0f);
    create_input(function->get_parameters()[2], v_shape, idx + 3.0f);
    create_input(function->get_parameters()[3], targetInputStaticShapes[1], idx + 4.0f);
    create_input(function->get_parameters()[4], v_init_shape, idx + 4.0f);
    create_input(function->get_parameters()[5], ov::Shape{targetInputStaticShapes[0][0]}, idx + 0.0f);
}

void ConcatSDPTest::prepare() {
    compile_model();
    inferRequest = compiledModel.create_infer_request();
    ASSERT_TRUE(inferRequest);
}

void ConcatSDPTest::reset() {
    for (auto&& state : inferRequest.query_state()) {
        state.reset();
    }
}

std::vector<ov::Tensor> ConcatSDPTest::run_test(std::shared_ptr<ov::Model> model) {
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
TEST_P(ConcatSDPTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ElementType inType;
    std::vector<InputShape> inputShapes;
    bool forceKVU8;
    bool hasShapeOf;
    bool isDiffKVHeadSize;
    std::tie(inType, inputShapes, forceKVU8, hasShapeOf, isDiffKVHeadSize) = this->GetParam();

    auto actualOutputs = run_test(function);
    if (!hasShapeOf) {
        CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 1);
        CheckNumberOfNodesWithType(compiledModel, "Concatenation", 0);
        CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
        CheckNumberOfNodesWithType(compiledModel, "Gather", 0);
    } else {
        // SimplifyGatherShapeOf will generate subgraph which contains Gather/Concat/Reorder, so could not check the number to confirm it's expected.
        // W/ or w/o fusion the SDPA name is the same, but if fused the output number should be 3(data+pastk+pastv)
        for (const auto& node : compiledModel.get_runtime_model()->get_ops()) {
            if (node->get_rt_info().find(ov::exec_model_info::LAYER_TYPE)->second.as<std::string>() == "ScaledDotProductAttention") {
                ASSERT_EQ(3, node->get_output_size()) << "ScaledDotProductAttention should be fused";
            }
        }
    }

    // the range of our result will exceed f16 max value and there may be 'inf'. In softmax, there is a step:
    //   v - max(v), if v is inf, the result of 'v-max(v)' will be nan
    // use f32 as reference
    if (inType == ElementType::f16) {
        configuration["INFERENCE_PRECISION_HINT"] = "f32";
    }

    auto expectedOutputs = run_test(functionRefs);
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 0);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}


}  // namespace test
}  // namespace ov
