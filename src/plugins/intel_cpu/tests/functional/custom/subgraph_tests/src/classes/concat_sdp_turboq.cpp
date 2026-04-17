// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "concat_sdp_turboq.hpp"

#include <random>

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/opsets/opset13_decl.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

// Fill tensor with deterministic pseudo-random values.
// Uses val as seed so different inputs get different but reproducible data.
template <typename IT, typename T>
static void fill_random_tbq(IT first, size_t n, T val) {
    std::mt19937 rng(static_cast<uint32_t>(val * 1000.0f + 42.0f));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        *first++ = static_cast<T>(dist(rng));
    }
}

std::string ConcatSDPTurboQTest::getTestCaseName(const testing::TestParamInfo<ConcatSDPTurboQTestParams>& obj) {
    const auto& [inType, inputShapes, kCacheMode, vCacheMode, rotationMode, isCausal] = obj.param;
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
    result << "K=" << kCacheMode << "_V=" << vCacheMode << "_";
    result << "Rot=" << rotationMode << "_";
    result << "Causal=" << (isCausal ? "true" : "false");
    return result.str();
}

void ConcatSDPTurboQTest::SetUp() {
    const auto& [inType, inputShapes, kCacheMode, vCacheMode, rotationMode, isCausal] = this->GetParam();
    m_kCacheMode = kCacheMode;
    m_vCacheMode = vCacheMode;
    m_rotationMode = rotationMode;
    targetDevice = ov::test::utils::DEVICE_CPU;

    auto is_tbq = [](const std::string& mode) {
        return mode == "tbq3" || mode == "tbq4" || mode == "tbq3_qjl" || mode == "tbq4_qjl";
    };
    auto is_polar = [](const std::string& mode) {
        return mode == "polar3" || mode == "polar4";
    };
    auto is_codec = [&](const std::string& mode) {
        return is_tbq(mode) || is_polar(mode);
    };

    // Mixing TurboQuant and PolarQuant between K and V is unsupported.
    if ((is_tbq(kCacheMode) && is_polar(vCacheMode)) || (is_polar(kCacheMode) && is_tbq(vCacheMode))) {
        GTEST_SKIP() << "Mixing TurboQuant and PolarQuant between K/V is not supported";
    }

    // Determine the "strongest" codec for threshold selection.
    // Both K and V codec/quant errors compound; asymmetric (one side none/u8) has lower error.
    auto is_lossy = [](const std::string& mode) {
        return mode == "u4" || mode == "tbq3" || mode == "tbq3_qjl" || mode == "polar3" || mode == "tbq4" ||
               mode == "tbq4_qjl" || mode == "polar4";
    };
    bool any_3bit = (kCacheMode == "tbq3" || kCacheMode == "tbq3_qjl" || kCacheMode == "polar3" ||
                     vCacheMode == "tbq3" || vCacheMode == "tbq3_qjl" || vCacheMode == "polar3");
    bool any_4bit =
        (kCacheMode == "tbq4" || kCacheMode == "tbq4_qjl" || kCacheMode == "polar4" || vCacheMode == "tbq4" ||
         vCacheMode == "tbq4_qjl" || vCacheMode == "polar4" || kCacheMode == "u4" || vCacheMode == "u4");
    bool both_lossy = is_lossy(kCacheMode) && is_lossy(vCacheMode);

    if (any_3bit) {
        rel_threshold = both_lossy ? 1.0f : 0.7f;
        abs_threshold = both_lossy ? 3.0f : 2.0f;
    } else if (any_4bit) {
        rel_threshold = both_lossy ? 0.8f : 0.5f;
        abs_threshold = both_lossy ? 3.0f : 1.5f;
    } else {
        // Both u8/none — low error.
        rel_threshold = 0.3f;
        abs_threshold = 1.0f;
    }

    if (inType == ElementType::bf16 || inType == ElementType::f16) {
        configuration.insert({"INFERENCE_PRECISION_HINT", ov::element::Type(inType).get_type_name()});
        // Reduced precision compounds with codec quantization error.
        rel_threshold *= 2.0f;
        abs_threshold *= 2.0f;
    }

    // Configure K cache.
    if (is_codec(m_kCacheMode)) {
        configuration["KEY_CACHE_CODEC"] = m_kCacheMode;
    } else if (m_kCacheMode == "u8" || m_kCacheMode == "u4" || m_kCacheMode == "f32") {
        configuration["KEY_CACHE_PRECISION"] = m_kCacheMode;
    }

    // Configure V cache.
    if (is_codec(m_vCacheMode)) {
        configuration["VALUE_CACHE_CODEC"] = m_vCacheMode;
    } else if (m_vCacheMode == "u8" || m_vCacheMode == "u4" || m_vCacheMode == "f32") {
        configuration["VALUE_CACHE_PRECISION"] = m_vCacheMode;
    }

    // Rotation mode is controlled via OV_TURBOQ_ROTATION env var at process start.
    init_input_shapes(inputShapes);
    ov::ParameterVector inputParams;
    // GQA support: 3 input shapes → [Q, KV_current, KV_past]
    //              2 input shapes → [Q=KV_current, KV_past] (legacy, no GQA)
    const bool gqa = inputDynamicShapes.size() >= 3;
    const auto& q_shape = inputDynamicShapes[0];
    const auto& kv_shape = gqa ? inputDynamicShapes[1] : inputDynamicShapes[0];
    const auto& past_shape = gqa ? inputDynamicShapes[2] : inputDynamicShapes[1];

    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, q_shape));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, kv_shape));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, kv_shape));
    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");

    // past K/V init
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, past_shape));
    auto var_k = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_shape, inType, "pastk"});
    auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_k);
    pastk->set_friendly_name("pastk_r");

    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, past_shape));
    auto var_v = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_shape, inType, "pastv"});
    auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[4], var_v);
    pastv->set_friendly_name("pastv_r");

    // beam_idx
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ElementType::i32, ov::PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    inputParams.push_back(beam_idx);

    auto gatherK =
        std::make_shared<ov::op::v8::Gather>(pastk, beam_idx, op::v0::Constant::create(ElementType::i32, {}, {0}));
    auto gatherV =
        std::make_shared<ov::op::v8::Gather>(pastv, beam_idx, op::v0::Constant::create(ElementType::i32, {}, {0}));
    auto concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherK, inputParams[1]}, 2);
    auto concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherV, inputParams[2]}, 2);

    // For GQA: Unsqueeze+Broadcast+Reshape to expand K/V heads to match Q.
    // This is the pattern the CPU plugin fuses into its internal SDPA GQA path.
    std::shared_ptr<ov::Node> sdpK = concatK;
    std::shared_ptr<ov::Node> sdpV = concatV;
    if (gqa) {
        auto H = q_shape[1].get_length();
        auto S = q_shape[3].get_length();
        auto Hk = kv_shape[1].get_length();
        auto group_size = H / Hk;

        auto make_gqa_broadcast = [&](const std::shared_ptr<ov::Node>& node) {
            // [B, Hk, L, S] -> [B, Hk, 1, L, S]
            auto unsqueezed =
                std::make_shared<ov::op::v0::Unsqueeze>(node, op::v0::Constant::create(ov::element::i32, {}, {2}));
            // [B, Hk, 1, L, S] -> [B, Hk, group_size, L, S]
            auto broadcasted = std::make_shared<ov::op::v3::Broadcast>(
                unsqueezed,
                op::v0::Constant::create(ov::element::i32,
                                         {5},
                                         std::vector<int32_t>{1, 1, static_cast<int32_t>(group_size), 1, 1}),
                ov::op::BroadcastType::BIDIRECTIONAL);
            // [B, Hk, group_size, L, S] -> [B, H, L, S]
            return std::make_shared<ov::op::v1::Reshape>(
                broadcasted,
                op::v0::Constant::create(ov::element::i32,
                                         {4},
                                         std::vector<int32_t>{0, static_cast<int32_t>(H), -1, static_cast<int32_t>(S)}),
                true);
        };
        sdpK = make_gqa_broadcast(concatK);
        sdpV = make_gqa_broadcast(concatV);
    }

    auto sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(inputParams[0], sdpK, sdpV, isCausal);
    sdp->set_friendly_name("mha");
    auto add = std::make_shared<ov::op::v1::Add>(sdp, op::v0::Constant::create(inType, {1}, {1.0f}));
    auto pastk_assign = std::make_shared<op::v6::Assign>(concatK, var_k);
    auto pastv_assign = std::make_shared<op::v6::Assign>(concatV, var_v);
    pastk_assign->set_friendly_name("pastk_w");
    pastv_assign->set_friendly_name("pastv_w");

    ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    SinkVector sinks{pastk_assign, pastv_assign};
    function = std::make_shared<ov::Model>(results, sinks, inputParams, "ConcatSDPTurboQ");
    targetDevice = ov::test::utils::DEVICE_CPU;
    functionRefs = function->clone();
    pass::Manager manager;
    manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    manager.run_passes(functionRefs);
}

void ConcatSDPTurboQTest::generate(int idx, const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto create_input = [this](std::shared_ptr<op::v0::Parameter> param, ov::Shape shape, float val) {
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
            fill_random_tbq(static_cast<float*>(t.data()), t.get_size(), val);
            inputs.insert({param, t});
        } else if (param->get_element_type() == element::f16) {
            ov::Tensor t{ov::element::f16, shape};
            fill_random_tbq(static_cast<ov::float16*>(t.data()), t.get_size(), val);
            inputs.insert({param, t});
        } else {
            ASSERT_TRUE(param->get_element_type() == element::bf16);
            ov::Tensor t{ov::element::bf16, shape};
            fill_random_tbq(static_cast<ov::bfloat16*>(t.data()), t.get_size(), val);
            inputs.insert({param, t});
        }
    };
    // GQA: 3 shapes → [Q, KV_current, KV_past]; 2 shapes → [Q=KV, KV_past]
    const bool gqa = targetInputStaticShapes.size() >= 3;
    const auto& q_shape = targetInputStaticShapes[0];
    const auto& kv_shape = gqa ? targetInputStaticShapes[1] : targetInputStaticShapes[0];
    const auto& past_shape = gqa ? targetInputStaticShapes[2] : targetInputStaticShapes[1];

    // q, k, v, pastk_init, pastv_init, beam_idx
    create_input(function->get_parameters()[0], q_shape, idx + 1.0f);
    create_input(function->get_parameters()[1], kv_shape, idx + 2.0f);
    create_input(function->get_parameters()[2], kv_shape, idx + 3.0f);
    create_input(function->get_parameters()[3], past_shape, idx + 4.0f);
    create_input(function->get_parameters()[4], past_shape, idx + 4.0f);
    create_input(function->get_parameters()[5], ov::Shape{q_shape[0]}, idx + 0.0f);
}

void ConcatSDPTurboQTest::prepare() {
    compile_model();
    inferRequest = compiledModel.create_infer_request();
    ASSERT_TRUE(inferRequest);
}

void ConcatSDPTurboQTest::reset() {
    for (auto&& state : inferRequest.query_state()) {
        state.reset();
    }
}

std::vector<ov::Tensor> ConcatSDPTurboQTest::run_test(std::shared_ptr<ov::Model> model) {
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

TEST_P(ConcatSDPTurboQTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, inputShapes, kCacheMode, vCacheMode, rotationMode, isCausal] = this->GetParam();
    auto actualOutputs = run_test(function);

    // Verify SDPA fusion occurred.
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 1);
    CheckNumberOfNodesWithType(compiledModel, "Concatenation", 0);

    // Use f32 reference for f16 (softmax inf handling).
    if (inType == ElementType::f16) {
        configuration["INFERENCE_PRECISION_HINT"] = "f32";
    }
    // Reference model runs without TBQ (decomposed SDPA, f32 precision).
    configuration.erase("KEY_CACHE_CODEC");
    configuration.erase("VALUE_CACHE_CODEC");
    configuration.erase("KEY_CACHE_PRECISION");
    configuration.erase("VALUE_CACHE_PRECISION");

    auto expectedOutputs = run_test(functionRefs);
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 0);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

}  // namespace test
}  // namespace ov
