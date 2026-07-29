// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "concat_sdp.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

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
    const auto& [inType, inputShapes, cacheCfg, hasShapeOf, headNumQ, headNumKV] = obj.param;
    std::ostringstream result;
    result << "IS=";
    for (const auto& shape : inputShapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << "TS=";
    for (const auto& shape : inputShapes) {
        result << "(";
        for (const auto& itr : shape.second) {
            result << ov::test::utils::vec2str(itr);
        }
        result << ")_";
    }
    result << "Prc=" << inType << "_";
    result << "Cfg=";
    if (cacheCfg.empty()) {
        result << "NONE";
    } else {
        for (const auto& [k, v] : cacheCfg) {
            result << k << "=" << v.as<std::string>() << "/";
        }
    }
    result << "_HasShapeOf=" << hasShapeOf;
    result << "_Hq=" << headNumQ << "_Hkv=" << headNumKV;
    return result.str();
}

void ConcatSDPTest::SetUp() {
    const auto& [inType, inputShapes, cacheCfg, hasShapeOf, headNumQ, headNumKV] = this->GetParam();
    m_cacheCfg = cacheCfg;
    m_hasShapeOf = hasShapeOf;
    m_headNumQ = headNumQ;
    m_headNumKV = headNumKV;
    OPENVINO_ASSERT(m_headNumQ % m_headNumKV == 0, "head_num_q must be divisible by head_num_kv");

    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP() << "Host has no native bf16 support.";
    }
    if (inType == ElementType::f16 && !ov::with_cpu_x86_avx512_core_fp16()) {
        GTEST_SKIP() << "Host has no native f16 support.";
    }

    targetDevice = ov::test::utils::DEVICE_CPU;
    configuration["INFERENCE_PRECISION_HINT"] = ov::element::Type(inType).get_type_name();
    for (const auto& kv : m_cacheCfg) {
        configuration[kv.first] = kv.second;
    }

    // Reference: CPU plugin with codec keys stripped — same reduction order, isolates
    // codec/quant noise. Loose abs for quant codecs covers near-zero values where rel
    // explodes; tbq bias grows with accumulated state across iters.
    auto has_value = [&](const std::string& key, const std::string& needle) {
        auto it = m_cacheCfg.find(key);
        return it != m_cacheCfg.end() && it->second.as<std::string>() == needle;
    };
    const bool is_u4 = has_value("KEY_CACHE_PRECISION", "u4") || has_value("VALUE_CACHE_PRECISION", "u4");
    const bool is_u8 = has_value("KEY_CACHE_PRECISION", "u8") || has_value("VALUE_CACHE_PRECISION", "u8");
    const bool is_tbq = has_value("KEY_CACHE_QUANT_ALG", "TURBO") ||
                        has_value("VALUE_CACHE_QUANT_ALG", "TURBO");
    rel_threshold = 1e-2F;
    abs_threshold = 1e-3F;
    if (is_u4 && is_tbq) {
        abs_threshold = 0.1F;
    } else if (is_u4) {
        abs_threshold = 0.08F;
    } else if (is_tbq) {
        abs_threshold = 0.09F;
    } else if (is_u8) {
        abs_threshold = 0.02F;
    }
    init_input_shapes(inputShapes);

    auto q_ps = inputDynamicShapes[0];
    auto kv_ps = q_ps;
    kv_ps[1] = m_headNumKV;
    auto past_ps = inputDynamicShapes[1];
    past_ps[1] = m_headNumKV;

    ov::ParameterVector inputParams;
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, q_ps));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, kv_ps));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, kv_ps));
    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, past_ps));
    auto var_k = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_ps, inType, "pastk"});
    auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_k);
    pastk->set_friendly_name("pastk_r");
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, past_ps));
    auto var_v = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_ps, inType, "pastv"});
    auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[4], var_v);
    pastv->set_friendly_name("pastv_r");

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ElementType::i32, ov::PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    inputParams.push_back(beam_idx);

    auto gatherK = std::make_shared<ov::op::v8::Gather>(
        pastk, beam_idx, ov::op::v0::Constant::create(ElementType::i32, {}, {0}));
    auto gatherV = std::make_shared<ov::op::v8::Gather>(
        pastv, beam_idx, ov::op::v0::Constant::create(ElementType::i32, {}, {0}));

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

    std::shared_ptr<ov::Node> k_for_sdp = concatK;
    std::shared_ptr<ov::Node> v_for_sdp = concatV;
    if (m_headNumQ != m_headNumKV) {
        const auto group_size = static_cast<int32_t>(m_headNumQ / m_headNumKV);
        const auto Hq = static_cast<int32_t>(m_headNumQ);
        const auto head_size = static_cast<int32_t>(q_ps[3].get_length());
        auto make_gqa_broadcast = [&](const std::shared_ptr<ov::Node>& node) {
            auto unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(
                node, ov::op::v0::Constant::create(ov::element::i32, {}, {2}));
            auto broadcasted = std::make_shared<ov::op::v3::Broadcast>(
                unsqueezed,
                ov::op::v0::Constant::create(ov::element::i32, {5},
                                              std::vector<int32_t>{1, 1, group_size, 1, 1}),
                ov::op::BroadcastType::BIDIRECTIONAL);
            return std::make_shared<ov::op::v1::Reshape>(
                broadcasted,
                ov::op::v0::Constant::create(ov::element::i32, {4},
                                              std::vector<int32_t>{0, Hq, -1, head_size}),
                true);
        };
        k_for_sdp = make_gqa_broadcast(concatK);
        v_for_sdp = make_gqa_broadcast(concatV);
    }

    auto sdp = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
        inputParams[0], k_for_sdp, v_for_sdp, false);
    sdp->set_friendly_name("mha");
    auto add = std::make_shared<ov::op::v1::Add>(sdp, ov::op::v0::Constant::create(inType, {1}, {1.0f}));
    auto pastk_assign = std::make_shared<ov::op::v6::Assign>(concatK, var_k);
    auto pastv_assign = std::make_shared<ov::op::v6::Assign>(concatV, var_v);
    pastk_assign->set_friendly_name("pastk_w");
    pastv_assign->set_friendly_name("pastv_w");

    ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    if (m_hasShapeOf) {
        results.push_back(std::make_shared<ov::op::v0::Result>(shapeof_k));
        results.push_back(std::make_shared<ov::op::v0::Result>(shapeof_v));
    }
    SinkVector sinks{pastk_assign, pastv_assign};
    function = std::make_shared<ov::Model>(results, sinks, inputParams, "ConcatSDP");
    functionRefs = function->clone();
    pass::Manager manager;
    manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    manager.run_passes(functionRefs);
}

void ConcatSDPTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const int idx = m_iter++;
    const auto& params = function->get_parameters();
    auto fill_gaussian = [](const std::shared_ptr<ov::op::v0::Parameter>& param, const ov::Shape& shape, int seed) {
        return utils::create_and_fill_tensor_normal_distribution(param->get_element_type(),
                                                                 shape,
                                                                 /*mean=*/0.0F,
                                                                 /*stddev=*/0.2F,
                                                                 seed);
    };
    auto fill_beam_idx = [](const std::shared_ptr<ov::op::v0::Parameter>& param, const ov::Shape& shape, int start) {
        ov::Tensor t{param->get_element_type(), shape};
        auto* p = static_cast<int*>(t.data());
        const auto size = shape[0];
        for (size_t i = 0; i < size; i++) {
            p[i] = (start + static_cast<int>(i)) % static_cast<int>(size);
        }
        return t;
    };

    auto q_shape = targetInputStaticShapes[0];
    auto kv_shape = q_shape;
    kv_shape[1] = static_cast<size_t>(m_headNumKV);
    auto past_shape = targetInputStaticShapes[1];
    past_shape[1] = static_cast<size_t>(m_headNumKV);
    past_shape[2] = (idx == 0) ? 0 : m_accum_L_q;
    m_accum_L_q = past_shape[2] + q_shape[2];

    inputs.insert({params[0], fill_gaussian(params[0], q_shape, idx + 1)});
    inputs.insert({params[1], fill_gaussian(params[1], kv_shape, idx + 2)});
    inputs.insert({params[2], fill_gaussian(params[2], kv_shape, idx + 3)});
    inputs.insert({params[3], fill_gaussian(params[3], past_shape, idx + 4)});
    inputs.insert({params[4], fill_gaussian(params[4], past_shape, idx + 5)});
    inputs.insert({params[5], fill_beam_idx(params[5], ov::Shape{q_shape[0]}, idx)});
}

// Compile `model` with `cfg`, run all iters via generate_inputs, return per-iter
// outputs. Updates `compiledModel` so TEST_P post-checks inspect the last run.
// Inputs keyed by original Parameter pointers; match compiled ports by friendly
// name. Deep-copy outputs.
std::vector<std::vector<ov::Tensor>>
ConcatSDPTest::run_test(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& cfg) {
    compiledModel = core->compile_model(model, targetDevice, cfg);
    auto req = compiledModel.create_infer_request();
    m_iter = 0;
    m_accum_L_q = 0;
    std::vector<std::vector<ov::Tensor>> all;
    for (const auto& shapes : targetStaticShapes) {
        generate_inputs(shapes);
        for (const auto& port : compiledModel.inputs()) {
            const auto& name = port.get_node()->get_friendly_name();
            for (const auto& [node, tensor] : inputs) {
                if (node->get_friendly_name() == name) {
                    req.set_tensor(port, tensor);
                    break;
                }
            }
        }
        req.infer();
        std::vector<ov::Tensor> outs;
        for (const auto& port : compiledModel.outputs()) {
            const auto& src = req.get_tensor(port);
            ov::Tensor c{src.get_element_type(), src.get_shape()};
            src.copy_to(c);
            outs.push_back(std::move(c));
        }
        all.push_back(std::move(outs));
    }
    return all;
}

void ConcatSDPTest::run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    // Reference: SDPA decomposed into matmul/softmax (no cache codec). Strip cache
    // config keys so reference runs full-precision; keeps quant noise on actual side only.
    auto ref_config = configuration;
    for (const auto& key : {"KEY_CACHE_PRECISION", "VALUE_CACHE_PRECISION",
                            "KEY_CACHE_QUANT_ALG", "VALUE_CACHE_QUANT_ALG"}) {
        ref_config.erase(key);
    }
    auto expected = run_test(functionRefs, ref_config);
    auto actual = run_test(function, configuration);
    for (size_t i = 0; i < actual.size(); ++i) {
        compare(expected[i], actual[i]);
    }
}

TEST_P(ConcatSDPTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    if (!m_hasShapeOf) {
        CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 1);
        CheckNumberOfNodesWithType(compiledModel, "Concatenation", 0);
        CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
        CheckNumberOfNodesWithType(compiledModel, "Gather", 0);
    } else {
        for (const auto& node : compiledModel.get_runtime_model()->get_ops()) {
            if (node->get_rt_info().find(ov::exec_model_info::LAYER_TYPE)->second.as<std::string>() ==
                "ScaledDotProductAttention") {
                ASSERT_EQ(3, node->get_output_size()) << "ScaledDotProductAttention should be fused";
            }
        }
    }
}

}  // namespace test
}  // namespace ov
