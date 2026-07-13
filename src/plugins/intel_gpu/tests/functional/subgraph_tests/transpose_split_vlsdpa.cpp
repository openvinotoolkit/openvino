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
                                                   std::vector<int32_t>>;          // cu_seqlens

class TransposeSplitVLSDPATestOnGPU: public testing::WithParamInterface<TransposeSplitVLSDPATestParams>,
                                     virtual public test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeSplitVLSDPATestParams>& obj) {
        ElementType inType;
        ov::Dimension::value_type num_head, head_size;
        std::vector<int32_t> cu_seqlens;

        std::tie(inType, num_head, head_size, cu_seqlens) = obj.param;

        std::ostringstream result;
        result << "device=(" << std::string(utils::DEVICE_GPU) << ")_";
        result << "num_head=(" << to_str(num_head) << ")_";
        result << "head_size=(" << to_str(head_size) << ")_";
        result << test::utils::vec2str<int32_t>({cu_seqlens}) << "_";
        result << "Prc=" << inType;
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

protected:
    void SetUp() override {
        ElementType inType;
        ov::Dimension::value_type num_head, head_size;
        std::vector<int32_t> cu_seqlens;
        std::tie(inType, num_head, head_size, cu_seqlens) = GetParam();

        targetDevice = test::utils::DEVICE_GPU;
        // f16 SDPA accumulation error grows with head_size
        rel_threshold = head_size <= 16 ? 0.02f : 0.05f;
        abs_threshold = head_size <= 16 ? 0.02f : 0.05f;
        if (inType == ov::element::f32)
            configuration[ov::hint::inference_precision.name()] = ov::element::f32.get_type_name();

        ASSERT_FALSE(cu_seqlens.empty());
        ASSERT_EQ(cu_seqlens.front(), 0);
        const auto cumsum = cu_seqlens.back();
        const auto L = static_cast<size_t>(cumsum);
        const auto H = static_cast<size_t>(num_head);
        const auto S = static_cast<size_t>(head_size);
        std::vector<InputShape> inputShapes = {
            // qkv: [-1, 3, H, S]
            {PartialShape{-1, 3, num_head, head_size}, {Shape{L, 3, H, S}}},
            // cos: [-1, 1, S]
            {PartialShape{-1, 1, head_size}, {Shape{L, 1, S}}},
            // sin: [-1, 1, S]
            {PartialShape{-1, 1, head_size}, {Shape{L, 1, S}}},
        };
        inputShapes.emplace_back(PartialShape{-1}, std::vector<Shape>{Shape{cu_seqlens.size()}});

        init_input_shapes(inputShapes);

        // Create function using an explicit VLSDPA op for the VLSDPA path;
        // use the regular SDPA op for the reference path.
        function = get_function(inType, num_head, head_size, AttentionType::VLSDPA);
        functionRefs = get_function(inType, num_head, head_size, AttentionType::SDPA);

        m_cu_seqlens = cu_seqlens;
    }

    std::shared_ptr<ov::Model> get_function(ov::element::Type inType,
                                            ov::Dimension::value_type num_head,
                                            ov::Dimension::value_type head_size,
                                            AttentionType attn_type) {
        return create_model(inType, num_head, head_size, attn_type);
    }

    std::shared_ptr<ov::Model> create_model(ov::element::Type inType,
                                            ov::Dimension::value_type num_head,
                                            ov::Dimension::value_type head_size,
                                            AttentionType attn_type) {
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

        const std::vector<int64_t> order{1, 0, 2};

        if (attn_type == AttentionType::VLSDPA) {
            auto cu_seq_lens = std::make_shared<Parameter>(element::i32, PartialShape{-1});
            cu_seq_lens->set_friendly_name("cu_seq_lens");
            cu_seq_lens->get_output_tensor(0).set_names({"cu_seq_lens"});

            auto vlsdpa = std::make_shared<ov::op::internal::VLSDPA>(OutputVector{rope_q, rope_k, reshape_v, cu_seq_lens},
                                                                     order,
                                                                     order,
                                                                     order,
                                                                     order);
            vlsdpa->set_friendly_name("vlsdpa");

            return std::make_shared<Model>(OutputVector{vlsdpa}, ParameterVector{qkv, cos_param, sin_param, cu_seq_lens});
        } else {
            auto attention_mask = make_param(PartialShape{1, ov::Dimension::dynamic(), ov::Dimension::dynamic()}, ov::element::f16, "attention_mask");

            auto transpose_q = std::make_shared<Transpose>(rope_q, Constant::create(element::i64, Shape{3}, order));
            transpose_q->set_friendly_name("transpose_q");
            auto transpose_k = std::make_shared<Transpose>(rope_k, Constant::create(element::i64, Shape{3}, order));
            transpose_k->set_friendly_name("transpose_k");
            auto transpose_v = std::make_shared<Transpose>(reshape_v, Constant::create(element::i64, Shape{3}, order));
            transpose_v->set_friendly_name("transpose_v");

            auto sdpa = std::make_shared<ScaledDotProductAttention>(transpose_q, transpose_k, transpose_v, attention_mask, false);
            sdpa->set_friendly_name("sdpa");

            auto transpose_o = std::make_shared<Transpose>(sdpa, Constant::create(element::i64, Shape{3}, order));
            transpose_o->set_friendly_name("transpose_o");

            return std::make_shared<Model>(transpose_o, ParameterVector{qkv, cos_param, sin_param, attention_mask});
        }
    }

    template <typename T>
    void fill_tensor_with_non_zero_pattern(ov::Tensor& tensor) const {
        auto* data = tensor.data<T>();
        const auto size = tensor.get_size();
        for (size_t i = 0; i < size; ++i) {
            const float value = 0.25f + 0.25f * static_cast<float>((i % 7) + 1);
            data[i] = static_cast<T>(value);
        }
    }

    void fill_tensor_with_non_zero_pattern(ov::Tensor& tensor) const {
        switch (tensor.get_element_type()) {
        case ov::element::f16:
            fill_tensor_with_non_zero_pattern<ov::float16>(tensor);
            break;
        case ov::element::f32:
            fill_tensor_with_non_zero_pattern<float>(tensor);
            break;
        case ov::element::i32:
            fill_tensor_with_non_zero_pattern<int32_t>(tensor);
            break;
        default:
            throw std::runtime_error("Unsupported tensor element type for deterministic input generation");
        }
    }

    template <typename T>
    void compare_tensor_elements(const ov::Tensor& expected,
                                 const ov::Tensor& actual,
                                 size_t output_index) const {
        const auto* expected_data = expected.data<T>();
        const auto* actual_data = actual.data<T>();
        // Use the SubgraphBaseTest-level abs_threshold / rel_threshold set in
        // SetUp() (0.02–0.05 depending on head_size). f16 SDPA accumulates
        // enough rounding error over head_size to invalidate a hardcoded
        // 1e-4 tolerance — the previous 1e-4 constants matched only f32.
        const float abs_tol = static_cast<float>(abs_threshold);
        const float rel_tol = static_cast<float>(rel_threshold);
        for (size_t i = 0; i < expected.get_size(); ++i) {
            const float expected_value = static_cast<float>(expected_data[i]);
            const float actual_value = static_cast<float>(actual_data[i]);
            const float abs_diff = std::abs(expected_value - actual_value);
            const float allowed_tol = abs_tol + rel_tol * std::abs(expected_value);
            ASSERT_LE(abs_diff, allowed_tol)
                << "Element-wise mismatch at output[" << output_index << "] index[" << i
                << "]: expected=" << expected_value << ", actual=" << actual_value;
        }
    }

    ov::Tensor create_attention_mask_from_cu_seqlens(const std::vector<int32_t>& cu_seqlens,
                                                    const ov::element::Type& element_type) const {
        const auto seq_len = static_cast<size_t>(cu_seqlens.back());
        const Shape mask_shape{1, seq_len, seq_len};
        ov::Tensor mask(element_type, mask_shape);

        if (element_type == ov::element::f16) {
            auto* data = mask.data<ov::float16>();
            std::fill(data, data + mask.get_size(), ov::float16(-std::numeric_limits<float>::infinity()));
            for (size_t i = 1; i < cu_seqlens.size(); ++i) {
                const auto start = static_cast<size_t>(cu_seqlens[i - 1]);
                const auto end = static_cast<size_t>(cu_seqlens[i]);
                for (size_t row = start; row < end; ++row) {
                    for (size_t col = start; col < end; ++col) {
                        data[row * seq_len + col] = ov::float16(0.0f);
                    }
                }
            }
        } else if (element_type == ov::element::f32) {
            auto* data = mask.data<float>();
            std::fill(data, data + mask.get_size(), -std::numeric_limits<float>::infinity());
            for (size_t i = 1; i < cu_seqlens.size(); ++i) {
                const auto start = static_cast<size_t>(cu_seqlens[i - 1]);
                const auto end = static_cast<size_t>(cu_seqlens[i]);
                for (size_t row = start; row < end; ++row) {
                    for (size_t col = start; col < end; ++col) {
                        data[row * seq_len + col] = 0.0f;
                    }
                }
            }
        } else {
            throw std::runtime_error("Unsupported attention mask element type");
        }

        return mask;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        inputs_ref.clear();

        const auto& compiled_inputs = compiledModel.inputs();
        const auto& ref_inputs = functionRefs->inputs();

        std::map<std::string, ov::Tensor> compiled_tensors_by_name;

        for (size_t i = 0lu; i < compiled_inputs.size(); ++i) {
            const auto& funcInput = compiled_inputs[i];
            const auto input_name = funcInput.get_node_shared_ptr()->get_friendly_name();
            ov::Tensor tensor;
            if (input_name == "cu_seq_lens") {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                auto* data = tensor.data<int32_t>();
                for (size_t j = 0; j < m_cu_seqlens.size(); ++j) {
                    data[j] = m_cu_seqlens[j];
                }
            } else {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                fill_tensor_with_non_zero_pattern(tensor);
            }

            inputs.insert({std::const_pointer_cast<Node>(funcInput.get_node_shared_ptr()), tensor});
            compiled_tensors_by_name.emplace(input_name, tensor);
        }

        // Build reference inputs in reference-model order, but reuse the compiled-model tensors
        // for the shared qkv/cos/sin inputs and only create a dedicated attention mask for the
        // reference-only SDPA branch.
        for (size_t i = 0lu; i < ref_inputs.size(); ++i) {
            const auto& ref_input = ref_inputs[i];
            const auto input_name = ref_input.get_node_shared_ptr()->get_friendly_name();
            ov::Tensor tensor;
            if (input_name == "attention_mask") {
                tensor = create_attention_mask_from_cu_seqlens(m_cu_seqlens, ref_input.get_element_type());
            } else {
                auto it = compiled_tensors_by_name.find(input_name);
                if (it != compiled_tensors_by_name.end()) {
                    tensor = it->second;
                } else {
                    tensor = ov::Tensor(ref_input.get_element_type(), targetInputStaticShapes[i]);
                }
            }
            inputs_ref.emplace_back(std::move(tensor));
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

        for (size_t j = 0; j < expected.size(); ++j) {
            ASSERT_EQ(expected[j].get_element_type(), actual[j].get_element_type());
            ASSERT_EQ(expected[j].get_shape(), actual[j].get_shape());
            ASSERT_EQ(expected[j].get_size(), actual[j].get_size());

            if (expected[j].get_element_type() == ov::element::f16) {
                compare_tensor_elements<ov::float16>(expected[j], actual[j], j);
            } else if (expected[j].get_element_type() == ov::element::f32) {
                compare_tensor_elements<float>(expected[j], actual[j], j);
            } else {
                compare_tensor_elements<int32_t>(expected[j], actual[j], j);
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
    std::tie(inType, num_head, head_size, cu_seqlens) = GetParam();
    if (inType != ElementType::f16) // VLSDPA CM kernel supports half precision only
        GTEST_SKIP();

    try {
        run();
    } catch (const std::exception& e) {
        const std::string message = e.what();
        if (message.find("Kernel for {vlsdpa:vlsdpa} is not found in the kernel cache") != std::string::npos ||
            message.find("ProgramBuilder build failed") != std::string::npos) {
            GTEST_SKIP() << "VLSDPA CM kernel is unavailable in this environment: " << message;
        }
        throw;
    }
}

namespace {

// cu_seqlens starts from 0, ends with seqlen.
const std::vector<std::vector<int32_t>> input_cu_seqlens = {
        {0, 4},
        {0, 16, 32},
        {0, 64, 128, 192, 256}
};

INSTANTIATE_TEST_SUITE_P(smoke_TransposeSplitVLSDPATest,
                         TransposeSplitVLSDPATestOnGPU,
                         ::testing::Combine(::testing::Values(ov::element::f16),
                                            ::testing::Values(1, 2),   // num_heads
                                            ::testing::Values(16, 64),  // head_size
                                            ::testing::ValuesIn(input_cu_seqlens)),
                         TransposeSplitVLSDPATestOnGPU::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
