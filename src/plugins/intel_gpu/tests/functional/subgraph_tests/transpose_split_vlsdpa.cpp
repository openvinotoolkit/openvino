// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/engine_configuration.hpp"

#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset13.hpp"
#include "ov_ops/vl_sdpa.hpp"

#include "openvino/util/log.hpp"
#include "intel_gpu/runtime/execution_config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

//=================================================================================
// Subgraph Topology:
// Parameter + Transpose + Split → RoPE + VLSDPA pattern
// (Testing Split axis optimization: Transpose+Split(axis=0) → Split(axis=1))
//
// This pattern is from Qwen-VL Vision Merger models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, etc.)
// where a combined QKV tensor needs to be split into separate Q, K, V tensors.
//
// Before optimization:
// Parameter[-1,3,H,S] → Transpose[3,-1,H,S] → Split(axis=0) → 3x[1,-1,H,S]
//   → Reshape[0][-1,H,S] → RoPE → VLSDPA
//   → Reshape[1][-1,H,S] → RoPE ↗
//   → Reshape[2][-1,H,S] -------↗
//
// After optimization:
// Parameter[-1,3,H,S] → Split(axis=1) → 3x[-1,1,H,S]
//   → Reshape[0][-1,H,S] → RoPE → VLSDPA
//   → Reshape[1][-1,H,S] → RoPE ↗
//   → Reshape[2][-1,H,S] -------↗
//
// The transformation preserves results because:
// - Old: Transpose [N,C,H,S→C,N,H,S] then Split axis=0 extracts each channel
// - New: Split axis=1 directly extracts each channel without transpose
// - Both produce the same 3 tensors of shape [-1,H,S] with identical data
//=================================================================================

namespace ov {
namespace test {
using namespace ov;
using namespace ov::opset13;
using namespace ov::intel_gpu;

enum class AttentionType {
    SDPA,    // ScaledDotProductAttention
    VLSDPA   // Vision Language SDPA (for Qwen-VL models)
};

using TransposeSplitVLSDPATestParams = std::tuple<ElementType,
                                                   ov::Dimension::value_type,     // num_head
                                                   ov::Dimension::value_type,     // head_size
                                                   std::vector<int32_t>,          // cu_seqlens
                                                   AttentionType>;                // attention type

class TransposeSplitVLSDPATestOnGPU: public testing::WithParamInterface<TransposeSplitVLSDPATestParams>,
                                     virtual public test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeSplitVLSDPATestParams>& obj) {
        ElementType inType;
        ov::Dimension::value_type num_head, head_size;
        std::vector<int32_t> cu_seqlens;
        AttentionType attn_type;

        std::tie(inType, num_head, head_size, cu_seqlens, attn_type) = obj.param;

        std::ostringstream result;
        result << "device=(" << std::string(utils::DEVICE_GPU) << ")_";
        result << "num_head=(" << to_str(num_head) << ")_";
        result << "head_size=(" << to_str(head_size) << ")_";
        result << test::utils::vec2str<int32_t>({cu_seqlens}) << "_";
        result << "Prc=" << inType << "_";
        result << (attn_type == AttentionType::SDPA ? "SDPA" : "VLSDPA");
        return result.str();
    }

    static std::shared_ptr<Parameter> make_param(const PartialShape& pshape,
                                                element::Type element_type,
                                                const std::string& name) {
        auto param = std::make_shared<Parameter>(element_type, pshape);
        param->set_friendly_name(name);
        param->get_output_tensor(0).set_names({name});
        return param;
    }

    static bool check_vl_sdpa_transformations(const ov::CompiledModel& compiled_model) {
        const std::vector<std::string> target_names {"cu_seq_lens", "cu_window_seqlens"};

        bool exists = false;
        for (auto &input : compiled_model.inputs()) {
            const auto& names = input.get_names();

            for (const auto& target : target_names) {
                exists |= (names.find(target) != names.end());
            }
        }

        return exists;
    }

protected:
    void SetUp() override {
        ElementType inType;
        ov::Dimension::value_type num_head, head_size;
        std::vector<int32_t> cu_seqlens;
        AttentionType attn_type;
        std::tie(inType, num_head, head_size, cu_seqlens, attn_type) = GetParam();

        targetDevice = test::utils::DEVICE_GPU;
        // f16 SDPA accumulation error grows with head_size
        rel_threshold = head_size <= 16 ? 0.02f : 0.05f;
        abs_threshold = head_size <= 16 ? 0.02f : 0.05f;
        if (inType == ov::element::f32)
            configuration[ov::hint::inference_precision.name()] = ov::element::f32.get_type_name();

        assert(cu_seqlens.front() == 0);
        const auto cumsum = cu_seqlens.back();
        const auto L = static_cast<size_t>(cumsum);
        const auto H = static_cast<size_t>(num_head);
        const auto S = static_cast<size_t>(head_size);
        const std::vector<InputShape> inputShapes = {
            // qkv: [-1, 3, H, S]
            {PartialShape{-1, 3, num_head, head_size}, {Shape{L, 3, H, S}}},
            // cos: [-1, 1, S]
            {PartialShape{-1, 1, head_size}, {Shape{L, 1, S}}},
            // sin: [-1, 1, S]
            {PartialShape{-1, 1, head_size}, {Shape{L, 1, S}}},
        };
        init_input_shapes(inputShapes);

        // Create function - with model_type_hint for VLSDPA, without for SDPA
        function = get_function(inType, num_head, head_size);
        if (attn_type == AttentionType::VLSDPA) {
            function->set_rt_info("QWenVL", "model_type_hint");
        }

        // Create reference without model_type_hint - will keep original Transpose+Split pattern
        // Template plugin will execute SDPA without GPU transformations
        functionRefs = get_function(inType, num_head, head_size);

        m_cu_seqlens = cu_seqlens;
        m_attn_type = attn_type;
    }

    std::shared_ptr<ov::Model> get_function(ov::element::Type inType,
                                            ov::Dimension::value_type num_head,
                                            ov::Dimension::value_type head_size) {
        return create_model(inType, num_head, head_size);
    }

    std::shared_ptr<ov::Model> create_model(ov::element::Type inType,
                                            ov::Dimension::value_type num_head,
                                            ov::Dimension::value_type head_size) {
        // Parameter with shape [-1, 3, num_head, head_size]
        auto qkv = make_param(PartialShape{ov::Dimension::dynamic(), 3, num_head, head_size},
                             inType, "qkv");

        // Transpose: [-1, 3, H, S] -> [3, -1, H, S]
        auto transpose_order = Constant::create(element::i64, Shape{4}, std::vector<int64_t>{1, 0, 2, 3});
        auto transpose = std::make_shared<Transpose>(qkv, transpose_order);
        transpose->set_friendly_name("transpose_qkv");

        // Split along axis 0: [3, -1, H, S] -> 3x [1, -1, H, S]
        auto split_axis = Constant::create(element::i64, Shape{}, std::vector<int64_t>{0});
        auto split = std::make_shared<Split>(transpose, split_axis, 3);
        split->set_friendly_name("split_qkv");

        // Reshape each split output: [1, -1, H, S] -> [-1, H, S]
        auto reshape_pattern = Constant::create(element::i64, Shape{3}, std::vector<int64_t>{-1, num_head, head_size});

        auto reshape_q = std::make_shared<Reshape>(split->output(0), reshape_pattern, false);
        reshape_q->set_friendly_name("reshape_q");

        auto reshape_k = std::make_shared<Reshape>(split->output(1), reshape_pattern, false);
        reshape_k->set_friendly_name("reshape_k");

        auto reshape_v = std::make_shared<Reshape>(split->output(2), reshape_pattern, false);
        reshape_v->set_friendly_name("reshape_v");

        // Real Qwen-VL RoPE pattern (GPTNEOX_3D style):
        //   input [-1, H, S]
        //   cos_param [-1, 1, S], sin_param [-1, 1, S]
        //
        //   rope(input) = input * cos + rotate_half(input) * sin
        //   rotate_half: [-x_right, x_left]  (split along head_size dim)
        auto cos_param = make_param(PartialShape{ov::Dimension::dynamic(), 1, head_size}, inType, "cos");
        auto sin_param = make_param(PartialShape{ov::Dimension::dynamic(), 1, head_size}, inType, "sin");

        auto apply_rope = [&](const std::shared_ptr<ov::Node>& x, const std::string& prefix) {
            // rotate_half: split at head_size/2, negate right half, concat [-right, left]
            auto neg_one = Constant::create(inType, Shape{1, 1, 1}, {-1.0f});
            auto vs = std::make_shared<ov::opset1::VariadicSplit>(
                x,
                Constant::create(element::i64, Shape{}, {2}),
                Constant::create(element::i64, Shape{2}, std::vector<int64_t>{head_size / 2, head_size / 2}));
            vs->set_friendly_name(prefix + "_vs");
            auto neg = std::make_shared<Multiply>(vs->output(1), neg_one);
            neg->set_friendly_name(prefix + "_neg");
            auto rotated = std::make_shared<ov::opset1::Concat>(ov::OutputVector{neg, vs->output(0)}, -1);
            rotated->set_friendly_name(prefix + "_rotated");

            auto cos_mul = std::make_shared<Multiply>(x, cos_param);
            cos_mul->set_friendly_name(prefix + "_cos_mul");
            auto sin_mul = std::make_shared<Multiply>(rotated, sin_param);
            sin_mul->set_friendly_name(prefix + "_sin_mul");
            auto rope_out = std::make_shared<ov::opset1::Add>(cos_mul, sin_mul);
            rope_out->set_friendly_name(prefix + "_rope");
            return rope_out;
        };

        auto rope_q = apply_rope(reshape_q, "q");
        auto rope_k = apply_rope(reshape_k, "k");

        // Use ScaledDotProductAttention (will be transformed to VLSDPA on GPU with model_type_hint)
        auto sdpa = std::make_shared<ScaledDotProductAttention>(rope_q, rope_k, reshape_v, false);
        sdpa->set_friendly_name("sdpa");

        auto model = std::make_shared<Model>(sdpa, ParameterVector{qkv, cos_param, sin_param});
        // NOTE: model_type_hint will be set in SetUp() only for VLSDPA case
        return model;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        inputs_ref.clear();
        const auto& funcInputs = compiledModel.inputs();

        // Use small range [-0.5, 0.5] to avoid NaN in f16 SDPA softmax with larger head sizes
        for (size_t i = 0lu; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            auto tensor = utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                        targetInputStaticShapes[i],
                                                        1.0f,   // range
                                                        -0.5f); // start_from
            inputs.insert({std::const_pointer_cast<Node>(funcInput.get_node_shared_ptr()), tensor});
            inputs_ref.emplace_back(tensor);
        }
    }

    std::vector<ov::Tensor> calculate_refs() override {
        if (is_report_stages) {
            std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is started"<< std::endl;
        }
        auto start_time = std::chrono::system_clock::now();

        // Use Template plugin for reference - it doesn't have TransposeSplitMatcher
        // This keeps the original Transpose+Split pattern intact
        auto outputs = ov::test::utils::infer_on_template(functionRefs, inputs_ref);

        if (is_report_stages) {
            auto end_time = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is finished successfully. Duration is " << duration.count() << "s" << std::endl;
        }
        return outputs;
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        ASSERT_EQ(expected.size(), function->get_results().size());
        const auto& results = function->get_results();
        for (size_t j = 0; j < results.size(); j++) {
            const auto result = results[j];
            for (size_t i = 0; i < result->get_input_size(); ++i) {
                utils::compare(expected[j], actual[j], abs_threshold, rel_threshold);
            }
        }
    }

private:
    std::vector<int32_t> m_cu_seqlens;
    AttentionType m_attn_type;
    ov::TensorVector inputs_ref;
};

TEST_P(TransposeSplitVLSDPATestOnGPU, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ElementType inType;
    ov::Dimension::value_type num_head, head_size;
    std::vector<int32_t> cu_seqlens;
    AttentionType attn_type;
    std::tie(inType, num_head, head_size, cu_seqlens, attn_type) = GetParam();
    if (inType != ElementType::f16) // VLSDPA CM kernel supports half precision only
        GTEST_SKIP();

    run();
}

namespace {

// cu_seqlens starts from 0, ends with seqlen.
const std::vector<std::vector<int32_t>> input_cu_seqlens = {
        {0, 16},
        {0, 16, 32},
        {0, 64, 128, 192, 256}
};

INSTANTIATE_TEST_SUITE_P(smoke_TransposeSplitVLSDPATest,
                         TransposeSplitVLSDPATestOnGPU,
                         ::testing::Combine(::testing::Values(ov::element::f16),
                                            ::testing::Values(1, 2),   // num_heads
                                            ::testing::Values(16, 64),  // head_size
                                            ::testing::ValuesIn(input_cu_seqlens),
                                            ::testing::Values(AttentionType::SDPA, AttentionType::VLSDPA)),
                         TransposeSplitVLSDPATestOnGPU::getTestCaseName);

}  // namespace


//=================================================================================
// Accuracy regression test: vl_sdpa fed by transpose-split-fused packed QKV.
//
// Topology: packed QKV [-1, 3, H, S] -> Transpose([1,0,2,3]) -> Split(axis=0) ->
//           3x Reshape[-1,H,S] -> per-branch Transpose({1,0,2}) ->
//           ScaledDotProductAttention(q, k, v, attn_mask) -> Transpose({1,0,2}).
//
// Two GPU transformations combine here on purpose so the windowed CM vl_sdpa path is
// actually exercised (the existing TransposeSplitVLSDPATestOnGPU above uses full attention and
// never reaches it):
//   1. TransposeSplitMatcher rewrites Transpose([1,0,2,3]) + Split(axis=0) into an
//      in-place Split(axis=1).
//   2. SDPAToVLSDPA (gated by the "QWenVL" model_type_hint) rewrites
//      ScaledDotProductAttention to the internal VLSDPA op, which dispatches the CM
//      vl_sdpa kernel and replaces the attention_mask input with cu_seqlens.
//
// Root cause exercised: the CM vl_sdpa online-softmax output accumulator rO was left
// uninitialized. The kernel zeroes it implicitly by scaling rO with the first block's
// max_comp = exp(-inf) = 0, but 0 * NaN == NaN, so stale NaN/Inf bits in the GRF
// registers backing rO leak straight into the output. head_size=64 spans 8 rO register
// tiles and reliably hits dirty registers on long sequences, so that case fails (and is
// run-to-run non-deterministic on real models); head_size=16 (2 tiles) stays clean.
//
// The reference (functionRefs) runs on the Template plugin, which has neither
// transformation: it keeps the contiguous Transpose+Split and computes windowed
// attention from an equivalent block-diagonal attention_mask.
//
// AttentionType::VLSDPA -> model_type_hint set -> CM vl_sdpa kernel (under test)
// AttentionType::SDPA   -> no hint             -> plain OCL SDPA on the same buffers
//                                                 (control: must always pass)
//=================================================================================
namespace {
using TransposeSplitVLSDPAAccuracyParams = std::tuple<ElementType,
                                                      ov::Dimension::value_type,    // num_head
                                                      ov::Dimension::value_type,    // head_size
                                                      std::vector<int32_t>,         // cu_seqlens
                                                      AttentionType>;               // attention impl

class TransposeSplitVLSDPAAccuracyTest : public testing::WithParamInterface<TransposeSplitVLSDPAAccuracyParams>,
                                         virtual public test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeSplitVLSDPAAccuracyParams>& obj) {
        ElementType inType;
        ov::Dimension::value_type num_head, head_size;
        std::vector<int32_t> cu_seqlens;
        AttentionType attn_impl;
        std::tie(inType, num_head, head_size, cu_seqlens, attn_impl) = obj.param;

        std::ostringstream result;
        result << "device=(" << std::string(utils::DEVICE_GPU) << ")_";
        result << "num_head=(" << num_head << ")_";
        result << "head_size=(" << head_size << ")_";
        result << test::utils::vec2str<int32_t>({cu_seqlens}) << "_";
        result << "Prc=" << inType << "_";
        result << (attn_impl == AttentionType::SDPA ? "SDPA" : "VLSDPA");
        return result.str();
    }

    static std::shared_ptr<Parameter> make_param(const PartialShape& pshape,
                                                 element::Type element_type,
                                                 const std::string& name) {
        auto param = std::make_shared<Parameter>(element_type, pshape);
        param->set_friendly_name(name);
        param->get_output_tensor(0).set_names({name});
        return param;
    }

    static bool check_vl_sdpa_transformations(const ov::CompiledModel& compiled_model) {
        const std::vector<std::string> target_names{"cu_seq_lens", "cu_window_seqlens"};
        bool exists = false;
        for (const auto& input : compiled_model.inputs()) {
            const auto& names = input.get_names();
            for (const auto& target : target_names) {
                exists |= (names.find(target) != names.end());
            }
        }
        return exists;
    }

protected:
    void SetUp() override {
        ElementType inType;
        ov::Dimension::value_type num_head, head_size;
        std::vector<int32_t> cu_seqlens;
        AttentionType attn_impl;
        std::tie(inType, num_head, head_size, cu_seqlens, attn_impl) = GetParam();

        targetDevice = test::utils::DEVICE_GPU;
        // f16 SDPA accumulation error grows with head_size.
        rel_threshold = head_size <= 16 ? 0.02f : 0.05f;
        abs_threshold = head_size <= 16 ? 0.02f : 0.05f;

        OPENVINO_ASSERT(cu_seqlens.front() == 0);
        const auto L = static_cast<size_t>(cu_seqlens.back());
        const auto H = static_cast<size_t>(num_head);
        const auto S = static_cast<size_t>(head_size);
        const std::vector<InputShape> inputShapes = {
            {PartialShape{-1, 3, num_head, head_size}, {Shape{L, 3, H, S}}},  // packed qkv
            {PartialShape{1, -1, -1}, {Shape{1, L, L}}},                      // attention_mask
        };
        init_input_shapes(inputShapes);

        function = get_function(inType, num_head, head_size, attn_impl);
        functionRefs = function->clone();

        m_cu_seqlens = cu_seqlens;
    }

    std::shared_ptr<ov::Model> get_function(ov::element::Type inType,
                                            ov::Dimension::value_type num_head,
                                            ov::Dimension::value_type head_size,
                                            AttentionType attn_impl) {
        auto qkv = make_param(PartialShape{ov::Dimension::dynamic(), 3, num_head, head_size}, inType, "qkv");

        // Transpose: [-1, 3, H, S] -> [3, -1, H, S]
        auto transpose_order = Constant::create(element::i64, Shape{4}, std::vector<int64_t>{1, 0, 2, 3});
        auto transpose = std::make_shared<Transpose>(qkv, transpose_order);
        transpose->set_friendly_name("transpose_qkv");

        // Split along axis 0: [3, -1, H, S] -> 3x [1, -1, H, S]
        auto split_axis = Constant::create(element::i64, Shape{}, std::vector<int64_t>{0});
        auto split = std::make_shared<Split>(transpose, split_axis, 3);
        split->set_friendly_name("split_qkv");

        // Reshape each split output: [1, -1, H, S] -> [-1, H, S]
        auto reshape_pattern = Constant::create(element::i64, Shape{3}, std::vector<int64_t>{-1, num_head, head_size});
        auto reshape_q = std::make_shared<Reshape>(split->output(0), reshape_pattern, false);
        auto reshape_k = std::make_shared<Reshape>(split->output(1), reshape_pattern, false);
        auto reshape_v = std::make_shared<Reshape>(split->output(2), reshape_pattern, false);
        reshape_q->set_friendly_name("reshape_q");
        reshape_k->set_friendly_name("reshape_k");
        reshape_v->set_friendly_name("reshape_v");

        // Per-branch Transpose [L, H, S] -> [H, L, S] so SDPA attends along the L axis.
        // SDPAToVLSDPA folds these transposes (order {1,0,2}) into the VLSDPA op.
        auto transpose_q = std::make_shared<Transpose>(reshape_q, Constant::create(element::i64, Shape{3}, m_order));
        auto transpose_k = std::make_shared<Transpose>(reshape_k, Constant::create(element::i64, Shape{3}, m_order));
        auto transpose_v = std::make_shared<Transpose>(reshape_v, Constant::create(element::i64, Shape{3}, m_order));
        transpose_q->set_friendly_name("transpose_q");
        transpose_k->set_friendly_name("transpose_k");
        transpose_v->set_friendly_name("transpose_v");

        auto attn_mask = make_param(PartialShape{1, -1, -1}, inType, "attention_mask");
        const bool causal = false;
        auto sdpa = std::make_shared<ScaledDotProductAttention>(transpose_q, transpose_k, transpose_v, attn_mask, causal);
        sdpa->set_friendly_name("sdpa");

        auto transpose_o = std::make_shared<Transpose>(sdpa, Constant::create(element::i64, Shape{3}, m_order));
        transpose_o->set_friendly_name("transpose_o");

        auto model = std::make_shared<Model>(transpose_o, ParameterVector{qkv, attn_mask});
        if (attn_impl == AttentionType::VLSDPA) {
            // Triggers SDPAToVLSDPA (CM vl_sdpa kernel) on GPU.
            model->set_rt_info("QWenVL", "model_type_hint");
        }
        return model;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        inputs_ref.clear();
        const auto& funcInputs = compiledModel.inputs();
        for (size_t i = 0lu; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            if (i == 0) {  // packed qkv
                // Small range [-0.5, 0.5] avoids NaN in f16 SDPA softmax with larger head sizes.
                auto tensor = utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                            targetInputStaticShapes[i],
                                                            1.0f,    // range
                                                            -0.5f);  // start_from
                inputs.insert({std::const_pointer_cast<Node>(funcInput.get_node_shared_ptr()), tensor});
                inputs_ref.emplace_back(tensor);
            } else {  // cu_seqlens (GPU VLSDPA) or attention_mask (Template reference / SDPA control)
                auto attn_mask = get_attention_mask<ov::float16>(m_cu_seqlens, ov::element::f16);
                auto cu_seqlens = get_cu_seqlens(m_cu_seqlens);
                if (check_vl_sdpa_transformations(compiledModel)) {
                    inputs.insert({std::const_pointer_cast<Node>(funcInput.get_node_shared_ptr()), cu_seqlens});
                } else {
                    inputs.insert({std::const_pointer_cast<Node>(funcInput.get_node_shared_ptr()), attn_mask});
                }
                inputs_ref.emplace_back(attn_mask);
            }
        }
    }

    std::vector<ov::Tensor> calculate_refs() override {
        // Template plugin lacks TransposeSplitMatcher/SDPAToVLSDPA, so it keeps the
        // contiguous Transpose+Split and computes windowed SDPA from the mask.
        return ov::test::utils::infer_on_template(functionRefs, inputs_ref);
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        ASSERT_EQ(expected.size(), function->get_results().size());
        for (size_t j = 0; j < expected.size(); j++) {
            utils::compare(expected[j], actual[j], abs_threshold, rel_threshold);
        }
    }

private:
    template <typename ET>
    ov::Tensor get_attention_mask(const std::vector<int32_t>& cu_seqlens, ov::element::Type_t inType) const {
        OPENVINO_ASSERT(cu_seqlens.front() == 0);
        const size_t hidden_states_size = cu_seqlens.back();

        // Block-diagonal attention mask matching the cu_seqlens windows.
        ov::Tensor attention_mask{inType, {1, hidden_states_size, hidden_states_size}};
        ET* attention_mask_data = attention_mask.data<ET>();
        std::fill_n(attention_mask_data, attention_mask.get_size(), -std::numeric_limits<ET>::infinity());

        for (size_t i = 1; i < cu_seqlens.size(); ++i) {
            const size_t start = cu_seqlens[i - 1];
            const size_t end = cu_seqlens[i];
            for (size_t row = start; row < end; ++row) {
                for (size_t col = start; col < end; ++col) {
                    attention_mask_data[row * hidden_states_size + col] = ET(0.0f);
                }
            }
        }
        return attention_mask;
    }

    ov::Tensor get_cu_seqlens(const std::vector<int32_t>& cu_seqlens) const {
        OPENVINO_ASSERT(cu_seqlens.front() == 0);
        ov::Tensor t_cu_seqlens = ov::Tensor(ov::element::i32, {cu_seqlens.size()});
        auto* ptr = static_cast<int32_t*>(t_cu_seqlens.data());
        for (size_t n = 0; n < cu_seqlens.size(); n++) {
            ptr[n] = cu_seqlens[n];
        }
        return t_cu_seqlens;
    }

    const std::vector<int64_t> m_order = {1, 0, 2};
    std::vector<int32_t> m_cu_seqlens;
    ov::TensorVector inputs_ref;
};

TEST_P(TransposeSplitVLSDPAAccuracyTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ElementType inType;
    ov::Dimension::value_type num_head, head_size;
    std::vector<int32_t> cu_seqlens;
    AttentionType attn_impl;
    std::tie(inType, num_head, head_size, cu_seqlens, attn_impl) = GetParam();
    if (inType != ElementType::f16)  // VLSDPA CM kernel supports half precision only.
        GTEST_SKIP();

    run();
}

// cu_seqlens starts from 0 and ends with the total sequence length.
const std::vector<std::vector<int32_t>> accuracy_cu_seqlens = {
    {0, 16},
    {0, 16, 32},
    {0, 64, 128, 192, 256},
    // Long single window (max_seq_len > 256) forces wg_size > 16 in the VLSDPA CM kernel
    // dispatch (need_wg_mapping = 1), mirroring the ~1024-token single-image vision sequence.
    {0, 1024},
};

INSTANTIATE_TEST_SUITE_P(smoke_TransposeSplitVLSDPAAccuracy,
                         TransposeSplitVLSDPAAccuracyTest,
                         ::testing::Combine(::testing::Values(ov::element::f16),
                                            ::testing::Values(1, 2),    // num_heads
                                            ::testing::Values(16, 64),  // head_size
                                            ::testing::ValuesIn(accuracy_cu_seqlens),
                                            ::testing::Values(AttentionType::SDPA, AttentionType::VLSDPA)),
                         TransposeSplitVLSDPAAccuracyTest::getTestCaseName);
}  // namespace

}  // namespace test
}  // namespace ov
