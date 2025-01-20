// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "transformations/sdpa_to_paged_attention/prev_sequence_length_pattern.hpp"
#include "transformations/sdpa_to_paged_attention/state_management_pattern.hpp"
#include "transformations/sdpa_to_paged_attention/total_sequence_length_pattern.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/print_model.hpp"

using namespace ov;
using namespace std;
using namespace testing;
using namespace ov::op;
using namespace ov::gen_pattern;

namespace {

// Constants and Parameters attributes:
auto el_type_i64 = std::pair<std::string, detail::AttrAny>({"element_type", "i64"});
auto el_type_i32 = std::pair<std::string, detail::AttrAny>({"element_type", "i32"});
auto el_type_f32 = std::pair<std::string, detail::AttrAny>({"element_type", "f32"});

// Convert ops attributes:
auto dest_type_i64 = std::pair<std::string, detail::AttrAny>({"destination_type", "i64"});
auto dest_type_f32 = std::pair<std::string, detail::AttrAny>({"destination_type", "f32"});
auto dest_type_f16 = std::pair<std::string, detail::AttrAny>({"destination_type", "f16"});

// Other attributes:
auto numpy_broadcast = std::pair<std::string, detail::AttrAny>({"auto_broadcast", "numpy"});
auto special_zero_true = std::pair<std::string, detail::AttrAny>({"special_zero", true});

auto single_val = [](int rank, float val) {
    return makeConst(element::f32, ov::Shape{std::vector<size_t>(rank, 1)}, {val});
};

ov::ParameterVector nodes_to_params(const ov::NodeVector& node_vec) {
    ov::ParameterVector params;
    params.reserve(node_vec.size());
    for (const auto& node : node_vec) {
        params.push_back(ov::as_type_ptr<v0::Parameter>(node));
    }
    return params;
}

enum QKV : int { Q = 0, K = 1, V = 2 };
vector<int> MOCK_VALUE = {1};

// original weights = 151936, attention_weights = 12288
#define WEIGHTS           1024
#define ATTENTION_WEIGHTS 512

class Qwen7bChatSDPA {
public:
    static std::shared_ptr<Node> gen_embeddings(const std::shared_ptr<Node>& input_ids) {
        auto view_reshape = makeOP<v1::Reshape>({input_ids, {-1, 0}}, {special_zero_true});
        auto input_ids_i64 = makeOP<v0::Convert>({view_reshape}, {dest_type_i64});

        auto weights = makeConst(element::u8, {WEIGHTS, 4096}, MOCK_VALUE);
        auto weights_fp16 = makeOP<v0::Convert>({weights}, {dest_type_f16});
        auto zero_point = makeConst(element::u8, {WEIGHTS, 1}, MOCK_VALUE);
        auto zero_point_fp16 = makeOP<v0::Convert>({zero_point}, {dest_type_f16});
        auto zero_point_subtract = makeOP<v1::Subtract>({weights_fp16, zero_point_fp16}, {numpy_broadcast});

        auto scale = makeConst(element::f16, {WEIGHTS, 1}, MOCK_VALUE);
        auto mul_scale = makeOP<v1::Multiply>({zero_point_subtract, scale}, {numpy_broadcast});
        auto fq_weights = makeOP<v0::Convert>({mul_scale}, {dest_type_f32});

        return makeOP<v8::Gather>({fq_weights, input_ids_i64, 0}, {{"batch_dims", 0}});
    }

    static std::shared_ptr<Node> gen_attention_weights() {
        auto weights = makeConst(element::u8, {ATTENTION_WEIGHTS, 4096}, MOCK_VALUE);
        auto weights_f16 = makeOP<v0::Convert>({weights}, {dest_type_f16});

        auto zero_points = makeConst(element::u8, {ATTENTION_WEIGHTS, 1}, MOCK_VALUE);
        auto zero_points_f16 = makeOP<v0::Convert>({zero_points}, {dest_type_f16});
        auto subtract = makeOP<v1::Subtract>({weights_f16, zero_points_f16}, {numpy_broadcast});

        auto scale = makeConst(element::f16, {ATTENTION_WEIGHTS, 1}, MOCK_VALUE);
        auto mul = makeOP<v1::Multiply>({subtract, scale}, {numpy_broadcast});
        return makeOP<v0::Convert>({mul}, {dest_type_f32});
    }

    static std::shared_ptr<Node> gen_qkv_proj(const std::shared_ptr<Node>& embeddings) {
        auto _const_0 = single_val(/*rank*/ 3, /*val*/ 2);
        auto pow = makeOP<v1::Power>({embeddings, _const_0}, {numpy_broadcast});
        auto mean = makeOP<v1::ReduceMean>({pow, {-1}}, {{"keep_dims", true}});

        auto _const_1 = single_val(/*rank*/ 3, /*val*/ 1);
        auto add = makeOP<v1::Add>({mean, _const_1}, {numpy_broadcast});
        auto sqrt = makeOP<v0::Sqrt>({add});

        auto _const_2 = single_val(/*rank*/ 3, /*val*/ 1);
        auto div = makeOP<v1::Divide>({_const_2, sqrt}, {numpy_broadcast, {"m_pythondiv", true}});
        auto mul_0 = makeOP<v1::Multiply>({embeddings, div}, {numpy_broadcast});

        auto _const_3 = makeConst(element::f32, {1, 1, 4096}, MOCK_VALUE);
        auto mul_1 = makeOP<v1::Multiply>({mul_0, _const_3}, {numpy_broadcast});
        auto attention_weights = gen_attention_weights();
        auto linear_matmul =
            makeOP<v0::MatMul>({mul_1, attention_weights}, {{"transpose_a", false}, {"transpose_b", true}});

        auto _const_4 = makeConst(element::f32, {1, 1, ATTENTION_WEIGHTS}, MOCK_VALUE);
        auto linear_add = makeOP<v1::Add>({linear_matmul, _const_4}, {numpy_broadcast});
        return makeOP<v1::VariadicSplit>({linear_add, 2, {4096, 4096, -1}});
    }

    static std::shared_ptr<Node> gen_cache(const std::shared_ptr<Node>& input_ids,
                                           const std::shared_ptr<Node>& beam_idx,
                                           const std::string& name) {
        auto shape_of = makeOP<v3::ShapeOf>({input_ids}, {{"output_type", "i64"}});
        auto gather = makeOP<v8::Gather>({shape_of, {0}, 0}, {{"batch_dims", 0}});
        auto concat = makeOP<v0::Concat>({gather, {0ll}, {32ll}, {128ll}}, {{"axis", 0}});
        auto init_to_read = makeOP<v1::Broadcast>({0.000000f, concat}, {{"mode", "numpy"}});
        auto cache = makeOP<v6::ReadValue>(
            {init_to_read},
            {{"variable_id", name}, {"variable_type", "f32"}, {"variable_shape", PartialShape{DYN, DYN, 32, 128}}});
        return makeOP<v8::Gather>({cache, beam_idx, 0}, {{"batch_dims", 0}});
    }

    static std::shared_ptr<Node> gen_current_len(const std::shared_ptr<Node>& input_ids) {
        auto shape_of = makeOP<v3::ShapeOf>({input_ids}, {{"output_type", "i64"}});
        return makeOP<v8::Gather>({shape_of, {1}, 0}, {{"batch_dims", 0}});
    }

    static std::shared_ptr<Node> gen_past_len(const std::shared_ptr<Node>& k_cache) {
        auto shape_of = makeOP<v3::ShapeOf>({k_cache}, {{"output_type", "i64"}});
        return makeOP<v8::Gather>({shape_of, {1}, 0}, {{"batch_dims", 0}});
    }

    static std::shared_ptr<Node> gen_total_len(const std::shared_ptr<Node>& cur_len,
                                               const std::shared_ptr<Node>& past_len) {
        return makeOP<v1::Add>({cur_len, past_len}, {numpy_broadcast});
    }

    static std::shared_ptr<Node> gen_rope(QKV idx,
                                          const std::shared_ptr<Node>& qkv_proj,
                                          const std::shared_ptr<Node>& head_size,
                                          const std::shared_ptr<Node>& sliced_sin,
                                          const std::shared_ptr<Node>& sliced_cos) {
        auto current_k = makeOP<v1::Reshape>({qkv_proj->output(idx), {0, 0, 32, 128}}, {special_zero_true});
        auto sliced_k = makeOP<v8::Slice>({current_k, {0}, head_size, {1}, {3}});
        auto mul_1 = makeOP<v1::Multiply>({sliced_k, sliced_cos}, {numpy_broadcast});

        auto reshape = makeOP<v1::Reshape>({sliced_k, {0, 0, 32, 2, 64}}, {special_zero_true});
        auto split_1 = makeOP<v1::Split>({reshape, -2}, {{"num_splits", 2}});
        auto list_unpack_1 = makeOP<v0::Squeeze>({split_1->output(1), -2});

        auto _const = single_val(/*rank*/ 4, /*val*/ 1);
        auto mul_2 = makeOP<v1::Multiply>({list_unpack_1, _const}, {numpy_broadcast});
        auto list_unpack_2 = makeOP<v0::Squeeze>({split_1->output(0), -2});
        auto concat = makeOP<v0::Concat>({mul_2, list_unpack_2}, {{"axis", -1}});

        auto mul_3 = makeOP<v1::Multiply>({concat, sliced_sin}, {numpy_broadcast});
        return makeOP<v1::Add>({mul_1, mul_3}, {numpy_broadcast});
    }

    static std::shared_ptr<Node> gen_rope_emb_sin(const std::shared_ptr<Node>& total_seq_len,
                                                  const std::shared_ptr<Node>& neg_mul,
                                                  std::shared_ptr<Node>& head_size,
                                                  element::Type model_precision) {
        auto sin = makeConst(model_precision, {1, 4096, 1, 128}, MOCK_VALUE);
        if (model_precision != element::f32) {
            sin = makeOP<v0::Convert>({sin}, {dest_type_f32});
        }
        auto sliced_sin_by_total = makeOP<v8::Slice>({sin, {0}, total_seq_len, {1}, {1}});
        auto rotary_emb_sin_shape = makeOP<v3::ShapeOf>({sliced_sin_by_total}, {{"output_type", "i64"}});
        head_size = makeOP<v8::Gather>({rotary_emb_sin_shape, {3}, 0}, {{"batch_dims", 0}});
        return makeOP<v8::Slice>({sliced_sin_by_total, neg_mul, {LLONG_MAX}, {1}, {1}});
    }

    static std::shared_ptr<Node> gen_rope_emb_cos(const std::shared_ptr<Node>& total_seq_len,
                                                  const std::shared_ptr<Node>& neg_mul,
                                                  element::Type model_precision) {
        auto cos = makeConst(model_precision, {1, 4096, 1, 128}, MOCK_VALUE);
        if (model_precision != element::f32) {
            cos = makeOP<v0::Convert>({cos}, {dest_type_f32});
        }
        auto sliced_cos_by_total = makeOP<v8::Slice>({cos, {0}, total_seq_len, {1}, {1}});
        return makeOP<v8::Slice>({sliced_cos_by_total, neg_mul, {LLONG_MAX}, {1}, {1}});
    }

    static std::shared_ptr<Node> neg_mul(const std::shared_ptr<Node>& current_seq_len) {
        return makeOP<v1::Multiply>({current_seq_len, {-1ll}}, {numpy_broadcast});
    }

    static std::shared_ptr<Node> gen_V(const std::shared_ptr<Node>& cache, const std::shared_ptr<Node>& qkv_proj) {
        auto v_current = makeOP<v1::Reshape>({qkv_proj->output(2), {0, 0, 32, 128}}, {special_zero_true});
        auto v_total = makeOP<v0::Concat>({cache, v_current}, {{"axis", 1}});
        return makeOP<v1::Transpose>({v_total, {0, 2, 1, 3}});
    }

    static std::shared_ptr<Node> gen_K(const std::shared_ptr<Node>& cache, const std::shared_ptr<Node>& rope_K) {
        auto full_k = makeOP<v0::Concat>({cache, rope_K}, {{"axis", 1}});
        return makeOP<v1::Transpose>({full_k, {0, 2, 1, 3}});
    }

    static std::shared_ptr<Node> gen_Q(const std::shared_ptr<Node>& past_seq_len_2,
                                       const std::shared_ptr<Node>& total_seq_len_2,
                                       const std::shared_ptr<Node>& rope_Q) {
        auto _const = makeConst(element::f32, {1, 32767, 1, 1}, MOCK_VALUE);
        auto slice = makeOP<v8::Slice>({_const, past_seq_len_2, total_seq_len_2, {1}, {1}});
        auto mul = makeOP<v1::Multiply>({rope_Q, slice}, {numpy_broadcast});
        return makeOP<v1::Transpose>({mul, {0, 2, 1, 3}});
    }

    static std::shared_ptr<Node> gen_total_seq_len_2(const std::shared_ptr<Node>& past_k_len,
                                                     const std::shared_ptr<Node>& rope_k) {
        auto shape_rope_k = makeOP<v3::ShapeOf>({rope_k}, {{"output_type", "i64"}});
        auto cur_len = makeOP<v8::Gather>({shape_rope_k, {1}, 0}, {{"batch_dims", 0}});
        return makeOP<v1::Add>({past_k_len, cur_len}, {numpy_broadcast});
    }

    static std::shared_ptr<Node> gen_past_seq_len_2(const std::shared_ptr<Node>& total_seq_len,
                                                    const std::shared_ptr<Node>& rope_q) {
        auto shape_rope_q = makeOP<v3::ShapeOf>({rope_q}, {{"output_type", "i64"}});
        auto cur_len = makeOP<v8::Gather>({shape_rope_q, {1}, 0}, {{"batch_dims", 0}});
        return makeOP<v1::Subtract>({total_seq_len, cur_len}, {numpy_broadcast});
    }

    static std::shared_ptr<Node> gen_attention_mask(const std::shared_ptr<Node>& Q_in,
                                                    const std::shared_ptr<Node>& attention_mask_in,
                                                    const std::shared_ptr<Node>& total_seq_len) {
        auto _const = makeConst(element::boolean, {1, 1, 8192, 8192}, MOCK_VALUE);
        auto shape_of_q = makeOP<v3::ShapeOf>({Q_in}, {{"output_type", "i64"}});
        auto gather = makeOP<v8::Gather>({shape_of_q, {2}, 0}, {{"batch_dims", 0}});
        auto sub_1 = makeOP<v1::Subtract>({total_seq_len, gather}, {numpy_broadcast});
        auto concat = makeOP<v0::Concat>({sub_1, {0ll}}, {{"axis", 0}});
        auto broadcast = makeOP<v3::Broadcast>({total_seq_len, {2}}, {{"mode", "numpy"}});
        auto slice = makeOP<v8::Slice>({_const, concat, broadcast, {1, 1}, {2, 3}});
        auto bitwise_not = makeOP<v13::BitwiseNot>({slice});

        auto _const_1 = single_val(/*rank*/ 4, /*val*/ 1);
        auto view_reshape = makeOP<v1::Reshape>({attention_mask_in, {0, 0}}, {special_zero_true});
        auto unsqueeze_0 = makeOP<v0::Unsqueeze>({view_reshape, 1});
        auto unsqueeze_1 = makeOP<v0::Unsqueeze>({unsqueeze_0, 2});
        auto convert_0 = makeOP<v0::Convert>({unsqueeze_1}, {dest_type_f32});

        auto _const_2 = single_val(/*rank*/ 4, /*val*/ 1);
        auto mul_1 = makeOP<v1::Multiply>({convert_0, _const_2}, {numpy_broadcast});
        auto sub_2 = makeOP<v1::Subtract>({_const_1, mul_1}, {numpy_broadcast});

        auto _const_3 = single_val(/*rank*/ 4, /*val*/ 1);
        auto mul_2 = makeOP<v1::Multiply>({sub_2, _const_3}, {numpy_broadcast});
        auto list_construct = makeOP<v0::Concat>({{1ll}, {1ll}, gather, {1ll}}, {{"axis", 0}});
        auto expand_broadcast = makeOP<v3::Broadcast>({mul_2, list_construct}, {{"mode", "bidirectional"}});
        return makeOP<v1::Select>({bitwise_not, -FLT_MAX, expand_broadcast}, {numpy_broadcast});
    }
};

class Qwen7bChatPA {
public:
    static std::shared_ptr<Node> gen_embeddings(const std::shared_ptr<Node>& input_ids) {
        auto weights = makeConst(element::u8, {WEIGHTS, 4096}, MOCK_VALUE);
        auto weights_fp16 = makeOP<v0::Convert>({weights}, {dest_type_f16});

        auto zero_point = makeConst(element::u8, {WEIGHTS, 1}, MOCK_VALUE);
        auto zero_point_fp16 = makeOP<v0::Convert>({zero_point}, {dest_type_f16});
        auto sub = makeOP<v1::Subtract>({weights_fp16, zero_point_fp16}, {numpy_broadcast});

        auto scale = makeConst(element::f16, {WEIGHTS, 1}, MOCK_VALUE);
        auto mul = makeOP<v1::Multiply>({sub, scale}, {numpy_broadcast});
        auto mul_fp32 = makeOP<v0::Convert>({mul}, {dest_type_f32});

        auto reshape_view = makeOP<v1::Reshape>({input_ids, {-1, 0}}, {special_zero_true});
        auto reshape_view_i64 = makeOP<v0::Convert>({reshape_view}, {dest_type_i64});
        return makeOP<v8::Gather>({mul_fp32, reshape_view_i64, 0}, {{"batch_dims", 0}});
    }

    static std::shared_ptr<Node> gen_qkv_proj(const std::shared_ptr<Node>& embeddings) {
        auto _const_0 = makeConst(element::f32, {1, 1, 1}, MOCK_VALUE);
        auto pow = makeOP<v1::Power>({embeddings, _const_0}, {numpy_broadcast});
        auto mean = makeOP<v1::ReduceMean>({pow, {-1}}, {{"keep_dims", true}});
        auto _const_1 = makeConst(element::f32, {1, 1, 1}, MOCK_VALUE);
        auto add_0 = makeOP<v1::Add>({mean, _const_1}, {numpy_broadcast});

        auto sqrt = makeOP<v0::Sqrt>({add_0});
        auto _const_2 = makeConst(element::f32, {1, 1, 1}, MOCK_VALUE);
        auto div = makeOP<v1::Divide>({_const_2, sqrt}, {numpy_broadcast, {"m_pythondiv", true}});
        auto mul_0 = makeOP<v1::Multiply>({embeddings, div}, {numpy_broadcast});

        auto _const_3 = makeConst(element::f32, {1, 1, 4096}, MOCK_VALUE);
        auto mul_1 = makeOP<v1::Multiply>({mul_0, _const_3}, {numpy_broadcast});

        auto _const_4 = makeConst(element::u8, {ATTENTION_WEIGHTS, 4096}, MOCK_VALUE);
        auto convert_0 = makeOP<v0::Convert>({_const_4}, {dest_type_f16});

        auto _const_5 = makeConst(element::u8, {ATTENTION_WEIGHTS, 1}, MOCK_VALUE);
        auto convert_1 = makeOP<v0::Convert>({_const_5}, {dest_type_f16});
        auto sub = makeOP<v1::Subtract>({convert_0, convert_1}, {numpy_broadcast});

        auto _const_6 = makeConst(element::f16, {ATTENTION_WEIGHTS, 1}, MOCK_VALUE);
        auto mul_2 = makeOP<v1::Multiply>({sub, _const_6}, {numpy_broadcast});
        auto convert_2 = makeOP<v0::Convert>({mul_2}, {dest_type_f32});
        auto matmul = makeOP<v0::MatMul>({mul_1, convert_2}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Constant_270 = makeConst(element::f32, {1, 1, ATTENTION_WEIGHTS}, MOCK_VALUE);
        auto add_1 = makeOP<v1::Add>({matmul, Constant_270}, {numpy_broadcast});

        return makeOP<v1::VariadicSplit>({add_1, 2, {4096, 4096, -1}});
    }

    static std::shared_ptr<Node> gen_rope(QKV idx,
                                          const std::shared_ptr<Node>& qkv_proj,
                                          const std::shared_ptr<Node>& head_size,
                                          const std::shared_ptr<Node>& sin,
                                          const std::shared_ptr<Node>& cos) {
        auto Q_or_K = makeOP<v1::Reshape>({qkv_proj->output(idx), {0, 0, 32, 128}}, {special_zero_true});
        auto sliced = makeOP<v8::Slice>({Q_or_K, {0}, head_size, {1}, {3}});
        auto mul_0 = makeOP<v1::Multiply>({sliced, sin}, {numpy_broadcast});

        auto reshape = makeOP<v1::Reshape>({sliced, {0, 0, 32, 2, 64}}, {special_zero_true});
        auto split = makeOP<v1::Split>({reshape, -2}, {{"num_splits", 2}});
        auto squeeze_0 = makeOP<v0::Squeeze>({split->output(1), -2});
        auto _const_0 = makeConst(element::f32, {1, 1, 1, 1}, {1.000000f});
        auto mul_1 = makeOP<v1::Multiply>({squeeze_0, _const_0}, {numpy_broadcast});

        auto squeeze_1 = makeOP<v0::Squeeze>({split->output(0), -2});
        auto concat = makeOP<v0::Concat>({mul_1, squeeze_1}, {{"axis", -1}});
        auto mul_2 = makeOP<v1::Multiply>({concat, cos}, {numpy_broadcast});
        return makeOP<v1::Add>({mul_0, mul_2}, {numpy_broadcast});
    }

    static std::shared_ptr<Node> gen_rope_emb_sin(const std::shared_ptr<Node>& max_context_len,
                                                  const std::shared_ptr<Node>& position_ids,
                                                  std::shared_ptr<Node>& head_size,
                                                  element::Type model_precision) {
        auto sin = makeConst(model_precision, {1, 4096, 1, 128}, MOCK_VALUE);
        if (model_precision != element::f32) {
            sin = makeOP<v0::Convert>({sin}, {dest_type_f32});
        }
        auto slice_sin = makeOP<v8::Gather>({sin, position_ids, 1}, {{"batch_dims", 0}});

        auto slice = makeOP<v8::Slice>({sin, {0}, max_context_len, {1}, {1}});
        auto shape_of = makeOP<opset3::ShapeOf>({slice}, {{"output_type", "i64"}});
        head_size = makeOP<v8::Gather>({shape_of, {3}, 0}, {{"batch_dims", 0}});

        return makeOP<v1::Reshape>({slice_sin, {-1, 1, 1, 128}}, {{"special_zero", false}});
    }

    static std::shared_ptr<Node> gen_rope_emb_cos(const std::shared_ptr<Node>& max_context_len,
                                                  const std::shared_ptr<Node>& position_ids,
                                                  element::Type model_precision) {
        auto cos = makeConst(model_precision, {1, 4096, 1, 128}, MOCK_VALUE);
        if (model_precision != element::f32) {
            cos = makeOP<v0::Convert>({cos}, {dest_type_f32});
        }
        auto slice = makeOP<v8::Gather>({cos, position_ids, 1}, {{"batch_dims", 0}});
        return makeOP<v1::Reshape>({slice, {-1, 1, 1, 128}}, {{"special_zero", false}});
    }

    static std::shared_ptr<Node> align_pa_layout(const std::shared_ptr<Node>& pa,
                                                 const std::shared_ptr<Node>& head_size) {
        auto shape = makeOP<v0::Concat>({{0ll}, {1ll}, {-1ll}, head_size}, {{"axis", 0}});
        auto reshaped = makeOP<v1::Reshape>({pa->output(0), shape}, {special_zero_true});
        return makeOP<v1::Transpose>({reshaped, {0, 2, 1, 3}});
    }

    static std::shared_ptr<Node> gen_current_len(const std::shared_ptr<Node>& rope_K) {
        auto shape_of = makeOP<opset3::ShapeOf>({rope_K}, {{"output_type", "i32"}});
        return makeOP<v8::Gather>({shape_of, {1}, 0ll}, {{"batch_dims", 0}});
    }

    static std::shared_ptr<Node> gen_past_len(const std::shared_ptr<Node>& input_ids,
                                              const std::shared_ptr<Node>& max_context_len) {
        auto shape_of = makeOP<opset3::ShapeOf>({input_ids}, {{"output_type", "i64"}});
        auto cur_len = makeOP<v8::Gather>({shape_of, 1ll, 0ll}, {{"batch_dims", 0}});
        auto cur_len_i32 = makeOP<v0::Convert>({cur_len}, {{"destination_type", "i32"}});

        auto past_len = makeOP<v1::Subtract>({max_context_len, cur_len_i32}, {numpy_broadcast});
        auto past_len_i32 = makeOP<v0::Convert>({past_len}, {{"destination_type", "i32"}});
        return makeOP<v1::Reshape>({past_len_i32, {1}}, {special_zero_true});
    }

    static std::shared_ptr<Node> gen_total_len(const std::shared_ptr<Node>& cur_len,
                                               const std::shared_ptr<Node>& past_len) {
        return makeOP<v1::Add>({past_len, cur_len}, {numpy_broadcast});
    }

    static std::shared_ptr<Node> gen_V(const std::shared_ptr<Node>& qkv_proj, std::shared_ptr<Node>& head_size) {
        auto current_V = makeOP<v1::Reshape>({qkv_proj->output(2), {0, 0, 32, 128}}, {special_zero_true});
        auto gather = makeOP<v8::Gather>({{0, 2, 1, 3}, {0, 2, 1, 3}, 0ll}, {{"batch_dims", 0}});
        auto transpose = makeOP<v1::Transpose>({current_V, gather});

        auto shape_of = makeOP<opset3::ShapeOf>({transpose}, {{"output_type", "i64"}});
        auto gather_2 = makeOP<v8::Gather>({shape_of, -1ll, 0ll}, {{"batch_dims", 0}});
        head_size = makeOP<v0::Unsqueeze>({gather_2, 0});

        return makeOP<v1::Reshape>({transpose, {0, -1}}, {special_zero_true});
    }

    static std::shared_ptr<Node> gen_K(const std::shared_ptr<Node>& rope_K) {
        auto gather = makeOP<v8::Gather>({{0, 2, 1, 3}, {0, 2, 1, 3}, 0ll}, {{"batch_dims", 0}});
        auto transpose = makeOP<v1::Transpose>({rope_K, gather});
        return makeOP<v1::Reshape>({transpose, {0, -1}}, {special_zero_true});
    }

    static std::shared_ptr<Node> gen_Q(const std::shared_ptr<Node>& total_seq_len,
                                       const std::shared_ptr<Node>& rope_Q) {
        auto _const_1 = makeConst(element::f32, {1, 32767, 1, 1}, MOCK_VALUE);
        auto shape_of = makeOP<opset3::ShapeOf>({rope_Q}, {{"output_type", "i32"}});
        auto current_seq_len = makeOP<v8::Gather>({shape_of, {1}, 0ll}, {{"batch_dims", 0}});
        auto past_seq_len = makeOP<v1::Subtract>({total_seq_len, current_seq_len}, {numpy_broadcast});

        auto slice = makeOP<v8::Slice>({_const_1, past_seq_len, total_seq_len, {1}, {1}});
        auto mul = makeOP<v1::Multiply>({rope_Q, slice}, {numpy_broadcast});
        auto transpose_1 = makeOP<v1::Transpose>({mul, {0, 2, 1, 3}});

        auto transpose_2 = makeOP<v1::Transpose>({transpose_1, {0, 2, 1, 3}});
        return makeOP<v1::Reshape>({transpose_2, {0, -1}}, {special_zero_true});
    }
};

}  // namespace

class SDPAToPATest : public TransformationTestsF, public ::testing::WithParamInterface<element::Type> {};

TEST_P(SDPAToPATest, SDPAToPA_Qwen7bChat_General) {
    const auto model_precision = GetParam();
    {
        // Inputs to SDPA transformer:
        auto beam_idx = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN}}, el_type_i64});
        auto position_ids = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN, DYN}}, el_type_i64});
        auto attention_mask = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN, DYN}}, el_type_i64});
        auto input_ids = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN, DYN}}, el_type_i64});
        ParameterVector params = nodes_to_params({position_ids, input_ids, attention_mask, beam_idx});

        beam_idx->output(0).add_names({"beam_idx"});
        position_ids->output(0).add_names({"position_ids"});
        attention_mask->output(0).add_names({"attention_mask"});
        input_ids->output(0).add_names({"input_ids"});

        // Embeddings processing:
        auto embeddings = Qwen7bChatSDPA::gen_embeddings(input_ids);
        auto qkv_proj = Qwen7bChatSDPA::gen_qkv_proj(embeddings);

        // KV cache:
        auto k_cache = Qwen7bChatSDPA::gen_cache(input_ids, beam_idx, "K_cache");
        auto v_cache = Qwen7bChatSDPA::gen_cache(input_ids, beam_idx, "V_cache");

        // Current/past/total Seq lengths calculation:
        auto current_seq_len = Qwen7bChatSDPA::gen_current_len(input_ids);
        auto past_seq_len = Qwen7bChatSDPA::gen_past_len(k_cache);
        auto total_seq_len = Qwen7bChatSDPA::gen_total_len(current_seq_len, past_seq_len);

        // RoPE emb sin/cos init:
        auto neg_cur_seq_len = Qwen7bChatSDPA::neg_mul(current_seq_len);
        auto head_size = shared_ptr<Node>();
        auto rope_emb_sin =
            Qwen7bChatSDPA::gen_rope_emb_sin(total_seq_len, neg_cur_seq_len, head_size, model_precision);
        auto rope_emb_cos = Qwen7bChatSDPA::gen_rope_emb_cos(total_seq_len, neg_cur_seq_len, model_precision);

        // RoPE for Q,K inputs:
        auto rope_q = Qwen7bChatSDPA::gen_rope(QKV::Q, qkv_proj, head_size, rope_emb_sin, rope_emb_cos);
        auto rope_k = Qwen7bChatSDPA::gen_rope(QKV::K, qkv_proj, head_size, rope_emb_sin, rope_emb_cos);

        // Lengths:
        auto total_seq_len_2 = Qwen7bChatSDPA::gen_total_seq_len_2(past_seq_len, rope_k);
        auto past_seq_len_2 = Qwen7bChatSDPA::gen_past_seq_len_2(total_seq_len_2, rope_q);

        // Q, K, V:
        auto Q = Qwen7bChatSDPA::gen_Q(past_seq_len_2, total_seq_len_2, rope_q);
        auto K = Qwen7bChatSDPA::gen_K(k_cache, rope_k);
        auto V = Qwen7bChatSDPA::gen_V(v_cache, qkv_proj);

        // Attention mask:
        auto attention_mask_to_sdpa = Qwen7bChatSDPA::gen_attention_mask(Q, attention_mask, total_seq_len_2);

        // SDPA:
        auto sdpa = makeOP<v13::ScaledDotProductAttention>({Q, K, V, attention_mask_to_sdpa}, {{"causal", false}});
        auto res = makeOP<v0::Result>({sdpa});

        model = std::make_shared<ov::Model>(OutputVector{res}, params);
        manager.register_pass<ov::pass::SDPAToPagedAttention>();
    }

    {
        // Inputs to PA transformer:
        auto max_context_len = makeOP<v0::Parameter>({}, {{"shape", PartialShape{}}, el_type_i32});
        auto block_indices_begins = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN}}, el_type_i32});
        auto block_indices = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN}}, el_type_i32});
        auto subsequence_begins = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN}}, el_type_i32});
        auto past_lens = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN}}, el_type_i32});
        auto value_cache_0 = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN, 32, 128}}, el_type_f32});
        auto key_cache_0 = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN, 32, 128}}, el_type_f32});
        auto input_ids = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN}}, el_type_i64});
        auto position_ids = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN}}, el_type_i64});
        auto params = nodes_to_params({max_context_len,
                                       block_indices_begins,
                                       block_indices,
                                       subsequence_begins,
                                       past_lens,
                                       value_cache_0,
                                       key_cache_0,
                                       input_ids,
                                       position_ids});

        // Inputs pre-processing:
        auto max_context_len_i64 = makeOP<v0::Convert>({max_context_len}, {dest_type_i64});
        auto max_context_len_aligned = makeOP<v1::Reshape>({max_context_len_i64, {1}}, {special_zero_true});
        auto input_ids_aligned = makeOP<v0::Unsqueeze>({input_ids, 1});
        auto position_ids_aligned = makeOP<v0::Unsqueeze>({position_ids, 1});

        // Embeddings processing:
        auto embeddings = Qwen7bChatPA::gen_embeddings(input_ids_aligned);
        auto qkv_proj = Qwen7bChatPA::gen_qkv_proj(embeddings);

        // RoPE emb sin/cos init:
        auto head_size = shared_ptr<Node>();
        auto rope_emb_sin =
            Qwen7bChatPA::gen_rope_emb_sin(max_context_len_aligned, position_ids_aligned, head_size, model_precision);
        auto rope_emb_cos =
            Qwen7bChatPA::gen_rope_emb_cos(max_context_len_aligned, position_ids_aligned, model_precision);

        // rope Q, K:
        auto rope_Q = Qwen7bChatPA::gen_rope(QKV::Q, qkv_proj, head_size, rope_emb_sin, rope_emb_cos);
        auto rope_K = Qwen7bChatPA::gen_rope(QKV::K, qkv_proj, head_size, rope_emb_sin, rope_emb_cos);

        // Current/past/total Seq lengths calculation:
        auto current_seq_len = Qwen7bChatPA::gen_current_len(rope_K);
        auto past_seq_len = Qwen7bChatPA::gen_past_len(input_ids_aligned, max_context_len);
        auto total_seq_len = Qwen7bChatPA::gen_total_len(current_seq_len, past_seq_len);

        // Q, K, V:
        shared_ptr<Node> head_size_2;
        auto Q = Qwen7bChatPA::gen_Q(total_seq_len, rope_Q);
        auto K = Qwen7bChatPA::gen_K(rope_K);
        auto V = Qwen7bChatPA::gen_V(qkv_proj, head_size_2);

        // Additional PA arguments:
        auto sliding_window = std::make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto alibi_slopes = std::make_shared<v0::Constant>(element::f32, Shape{0});
        auto scale = std::make_shared<v0::Constant>(element::f32, Shape{}, MOCK_VALUE);

        // PagedAttention:
        auto pa = std::make_shared<op::PagedAttentionExtension>(OutputVector{Q,
                                                                             K,
                                                                             V,
                                                                             key_cache_0,
                                                                             value_cache_0,
                                                                             past_lens,
                                                                             subsequence_begins,
                                                                             block_indices,
                                                                             block_indices_begins,
                                                                             scale,
                                                                             sliding_window,
                                                                             alibi_slopes,
                                                                             max_context_len});
        pa->set_out_type(0, element::i64);
        auto pa_aligned = Qwen7bChatPA::align_pa_layout(pa, head_size_2);
        auto res = makeOP<v0::Result>({pa_aligned});

        model_ref = std::make_shared<ov::Model>(OutputVector{res}, params);
    }
    // TODO: align precisions, check the copying of "fuse_names" attr in SDPAToPagedAttention
    // checking the graph structure and names, other checks are temporarily disabled:
    comparator.disable(FunctionsComparator::PRECISIONS);
    disable_rt_info_check();
}

TEST_P(SDPAToPATest, SDPAToPA_Qwen7bChat_TotalSequenceLengthPattern) {
    {
        // Inputs to SDPA transformer:
        auto beam_idx = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN}}, el_type_i64});
        auto input_ids = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN, DYN}}, el_type_i64});
        ParameterVector params = nodes_to_params({input_ids, beam_idx});

        // K cache
        auto k_cache = Qwen7bChatSDPA::gen_cache(input_ids, beam_idx, "K_cache");

        // Current/past/total Seq lengths calculation:
        auto current_len = Qwen7bChatSDPA::gen_current_len(input_ids);
        auto past_len = Qwen7bChatSDPA::gen_past_len(k_cache);
        auto total_len = Qwen7bChatSDPA::gen_total_len(current_len, past_len);
        auto result = std::make_shared<v0::Result>(total_len);

        // Expected that these Nodes to be created inside SDPAToPagedAttention
        auto new_input_ids = std::make_shared<v0::Parameter>(element::i64, PartialShape{DYN});
        auto axis = v0::Constant::create(element::i32, Shape{}, {1});
        auto aligned_input_ids = std::make_shared<v0::Unsqueeze>(new_input_ids, axis);

        input_ids->output(0).replace(aligned_input_ids);
        auto max_context_len = std::make_shared<v0::Parameter>(element::i32, PartialShape{});
        max_context_len->output(0).set_names({"max_context_len"});
        auto position_ids = std::make_shared<v0::Parameter>(element::i64, PartialShape{DYN});
        position_ids->output(0).set_names({"position_ids"});

        params.push_back(max_context_len);
        params.push_back(new_input_ids);

        // Model and Transformations:
        model = std::make_shared<ov::Model>(ResultVector{result}, params);
        manager.register_pass<pass::PrevSequenceLengthPattern>(aligned_input_ids, max_context_len, position_ids);
        manager.register_pass<pass::TotalSequenceLengthPatternQwen>(max_context_len);
    }

    {
        // Inputs to PA transformer:
        auto max_context_len = makeOP<v0::Parameter>({}, {{"shape", PartialShape{}}, el_type_i32});
        auto params = nodes_to_params({max_context_len});

        // Inputs pre-processing:
        auto max_context_len_i64 = makeOP<v0::Convert>({max_context_len}, {dest_type_i64});
        auto max_context_len_aligned = makeOP<v1::Reshape>({max_context_len_i64, {1}}, {special_zero_true});

        auto result = std::make_shared<v0::Result>(max_context_len_aligned);
        model_ref = std::make_shared<ov::Model>(ResultVector{result}, params);
    }
    // TODO: align precisions, check the copying of "fuse_names" attr in SDPAToPagedAttention
    // checking the graph structure and names, other checks are temporarily disabled:
    comparator.disable(FunctionsComparator::PRECISIONS);
    disable_result_friendly_names_check();
    disable_rt_info_check();
}

static std::shared_ptr<ov::Node> make_param(const PartialShape& pshape,
                                            element::Type element_type,
                                            const std::string& name) {
    auto param = makeOP<v0::Parameter>({}, {{"shape", pshape}, {"element_type", element_type}});
    param->set_friendly_name(name);
    param->get_output_tensor(0).set_names({name});
    return param;
}

// TODO: split the models in blocks the way it's done for Qwen and make the code not to be such a clutter
// TODO: write a test for StateManagementPattern only (because changes for Alibi are inside it)
// TODO: align precisions, check the copying of "fuse_names" attr in SDPAToPagedAttention
// checking the graph structure and names, other checks are temporarily disabled:
TEST_P(SDPAToPATest, SDPAToPA_Baichuan2_13b_General) {
    {
        auto beam_idx = make_param(PartialShape{DYN}, element::i32, "beam_idx");
        auto position_ids = make_param(PartialShape{DYN, DYN}, element::i64, "position_ids");
        auto attention_mask = make_param(PartialShape{DYN, DYN}, element::i64, "attention_mask");
        auto input_ids = make_param(PartialShape{DYN, DYN}, element::i64, "input_ids");

        // gen_embeddings() {
        auto ShapeOf5 = makeOP<v3::ShapeOf>({beam_idx}, {{"output_type", "i64"}});
        auto Gather8 = makeOP<v8::Gather>({ShapeOf5, {0ll}, 0ll}, {{"batch_dims", 0}});
        auto Concat12 = makeOP<v0::Concat>({Gather8, {40ll}, {0ll}, {128ll}}, {{"axis", 0}});
        auto Broadcast13 = makeOP<v3::Broadcast>({0.0f, Concat12}, {{"mode", "numpy"}});
        auto Constant18 = makeConst(element::u8, ov::Shape({125696, 5120}), MOCK_VALUE);
        auto Convert19 = makeOP<opset1::Convert>({Constant18}, {{"destination_type", "f16"}});
        auto Constant20 = makeConst(element::u8, ov::Shape({125696, 1}), MOCK_VALUE);
        auto Convert21 = makeOP<opset1::Convert>({Constant20}, {{"destination_type", "f16"}});
        auto Subtract22 = makeOP<opset1::Subtract>({Convert19, Convert21}, {{"auto_broadcast", "numpy"}});
        auto Constant23 = makeConst(element::f16, ov::Shape({125696, 1}), MOCK_VALUE);
        auto Multiply24 = makeOP<opset1::Multiply>({Subtract22, Constant23}, {{"auto_broadcast", "numpy"}});
        auto Convert25 = makeOP<opset1::Convert>({Multiply24}, {{"destination_type", "f32"}});
        auto Convert26 = makeOP<opset1::Convert>({input_ids}, {{"destination_type", "i32"}});
        auto Gather28 = makeOP<opset8::Gather>({Convert25, Convert26, 0}, {{"batch_dims", 0}});
        //}

        auto Constant29 = makeConst(element::f32, ov::Shape({1, 1, 5120}), MOCK_VALUE);
        auto Constant30 = makeConst(element::f32, ov::Shape({1, 1, 1}), {1.0f});
        auto Constant31 = makeConst(element::f32, ov::Shape({1, 1, 1}), {2.0f});
        auto Power32 = makeOP<opset1::Power>({Gather28, Constant31}, {{"auto_broadcast", "numpy"}});
        auto ReduceMean34 = makeOP<opset1::ReduceMean>({Power32, {-1}}, {{"keep_dims", true}});
        auto Constant35 = makeConst(element::f32, ov::Shape({1, 1, 1}), {0.000001f});
        auto Add36 = makeOP<opset1::Add>({ReduceMean34, Constant35}, {{"auto_broadcast", "numpy"}});
        auto Sqrt37 = makeOP<opset1::Sqrt>({Add36});
        auto Divide38 =
            makeOP<opset1::Divide>({Constant30, Sqrt37}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Multiply39 = makeOP<opset1::Multiply>({Gather28, Divide38}, {{"auto_broadcast", "numpy"}});
        auto Multiply40 = makeOP<opset1::Multiply>({Constant29, Multiply39}, {{"auto_broadcast", "numpy"}});

        // gen_attention_weights() {
        auto Constant41 = makeConst(element::u8, ov::Shape({15360, 5120}), MOCK_VALUE);
        auto Convert42 = makeOP<opset1::Convert>({Constant41}, {{"destination_type", "f16"}});
        auto Constant43 = makeConst(element::u8, ov::Shape({15360, 1}), MOCK_VALUE);
        auto Convert44 = makeOP<opset1::Convert>({Constant43}, {{"destination_type", "f16"}});
        auto Subtract45 = makeOP<opset1::Subtract>({Convert42, Convert44}, {{"auto_broadcast", "numpy"}});
        auto Constant46 = makeConst(element::f16, ov::Shape({15360, 1}), MOCK_VALUE);
        auto Multiply47 = makeOP<opset1::Multiply>({Subtract45, Constant46}, {{"auto_broadcast", "numpy"}});
        auto Convert48 = makeOP<opset1::Convert>({Multiply47}, {{"destination_type", "f32"}});
        //}

        auto MatMul49 =
            makeOP<opset1::MatMul>({Multiply40, Convert48}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Reshape51 = makeOP<opset1::Reshape>({MatMul49, {0, 0, 3, 5120}}, {{"special_zero", true}});
        auto Unsqueeze53 = makeOP<opset1::Unsqueeze>({Reshape51, 0});
        auto Squeeze55 = makeOP<opset1::Squeeze>({Unsqueeze53, {0}});
        auto Transpose57 = makeOP<opset1::Transpose>({Squeeze55, {2, 0, 1, 3}});

        // Q
        auto Gather58 = makeOP<opset8::Gather>({Transpose57, 0, 0}, {{"batch_dims", 0}});
        auto Reshape60 = makeOP<opset1::Reshape>({Gather58, {0, 0, 40, 128}}, {{"special_zero", true}});
        auto Transpose62 = makeOP<opset1::Transpose>({Reshape60, {0, 2, 1, 3}});

        auto ReadValue63 = makeOP<opset6::ReadValue>({Broadcast13},
                                                     {{"variable_id", "varid_2"},
                                                      {"variable_type", "f32"},
                                                      {"variable_shape", PartialShape{DYN, 40, DYN, 128}}});
        auto Gather65 = makeOP<opset8::Gather>({ReadValue63, beam_idx, 0}, {{"batch_dims", 0}});

        // K
        auto Gather67 = makeOP<opset8::Gather>({Transpose57, 1, 0}, {{"batch_dims", 0}});
        auto Reshape69 = makeOP<opset1::Reshape>({Gather67, {0, 0, 40, 128}}, {{"special_zero", true}});
        auto Transpose71 = makeOP<opset1::Transpose>({Reshape69, {0, 2, 1, 3}});
        auto Concat72 = makeOP<opset1::Concat>({Gather65, Transpose71}, {{"axis", 2}});

        auto ReadValue73 = makeOP<opset6::ReadValue>({Broadcast13},
                                                     {{"variable_id", "varid_3"},
                                                      {"variable_type", "f32"},
                                                      {"variable_shape", PartialShape{DYN, 40, DYN, 128}}});
        auto Gather75 = makeOP<opset8::Gather>({ReadValue73, beam_idx, 0}, {{"batch_dims", 0}});

        // V
        auto Gather77 = makeOP<opset8::Gather>({Transpose57, 2, 0}, {{"batch_dims", 0}});
        auto Reshape79 = makeOP<opset1::Reshape>({Gather77, {0, 0, 40, 128}}, {{"special_zero", true}});
        auto Transpose81 = makeOP<opset1::Transpose>({Reshape79, {0, 2, 1, 3}});
        auto Concat82 = makeOP<opset1::Concat>({Gather75, Transpose81}, {{"axis", 2}});

        auto Constant83 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {1.000000f});
        auto Convert85 = makeOP<opset1::Convert>({attention_mask}, {{"destination_type", "f32"}});
        auto Unsqueeze86 = makeOP<opset1::Unsqueeze>({Convert85, 2});
        auto Unsqueeze87 = makeOP<opset1::Unsqueeze>({Convert85, 1});
        auto Multiply88 = makeOP<opset1::Multiply>({Unsqueeze86, Unsqueeze87}, {{"auto_broadcast", "numpy"}});
        auto Constant89 = makeConst(element::f32, ov::Shape({1, 1, 1}), {0.000000f});
        auto Greater90 = makeOP<opset1::Greater>({Multiply88, Constant89}, {{"auto_broadcast", "numpy"}});
        auto ShapeOf91 = makeOP<opset3::ShapeOf>({Greater90}, {{"output_type", "i32"}});
        auto Gather94 = makeOP<opset8::Gather>({ShapeOf91, 1, 0}, {{"batch_dims", 0}});
        auto Range96 = makeOP<opset4::Range>({0, Gather94, 1}, {{"output_type", "i32"}});
        auto Unsqueeze97 = makeOP<opset1::Unsqueeze>({Range96, 0});
        auto Unsqueeze98 = makeOP<opset1::Unsqueeze>({Range96, 1});
        auto LessEqual99 = makeOP<opset1::LessEqual>({Unsqueeze97, Unsqueeze98}, {{"auto_broadcast", "numpy"}});
        auto Constant100 = makeConst(element::boolean, ov::Shape({}), {0});
        auto Select101 = makeOP<opset1::Select>({LessEqual99, Greater90, Constant100}, {{"auto_broadcast", "numpy"}});
        auto Subtract102 = makeOP<opset1::Subtract>({Unsqueeze86, Unsqueeze87}, {{"auto_broadcast", "numpy"}});
        auto Constant103 = makeConst(element::f32, ov::Shape({1, 1, 1}), {0.000000f});
        auto Equal104 = makeOP<opset1::Equal>({Subtract102, Constant103}, {{"auto_broadcast", "numpy"}});
        auto LogicalAnd105 = makeOP<opset1::LogicalAnd>({Select101, Equal104}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze106 = makeOP<opset1::Unsqueeze>({LogicalAnd105, 1});
        auto ShapeOf107 = makeOP<opset3::ShapeOf>({MatMul49}, {{"output_type", "i64"}});
        auto Gather110 = makeOP<opset8::Gather>({ShapeOf107, {0}, 0}, {{"batch_dims", 0}});
        auto Constant112 = makeConst(element::f32,
                                     ov::Shape({40, 4096, 4096}),
                                     MOCK_VALUE);  // TODO: there can be an error due to fake alibi slopes
        auto Gather116 = makeOP<opset8::Gather>({ShapeOf107, {1}, 0}, {{"batch_dims", 0}});
        auto ShapeOf117 = makeOP<opset3::ShapeOf>({Gather65}, {{"output_type", "i64"}});
        auto Gather120 = makeOP<opset8::Gather>({ShapeOf117, {2}, 0}, {{"batch_dims", 0}});
        auto Add121 = makeOP<opset1::Add>({Gather116, Gather120}, {{"auto_broadcast", "numpy"}});
        auto Broadcast123 = makeOP<opset3::Broadcast>({Add121, {2}}, {{"mode", "numpy"}});
        auto Slice126 =
            makeOP<opset8::Slice>({Constant112, {0, 0}, Broadcast123, {1, 1}, {1, 2}});  // the very slice we insert
        auto ShapeOf127 = makeOP<opset3::ShapeOf>({Slice126}, {{"output_type", "i64"}});
        auto Gather130 = makeOP<opset8::Gather>({ShapeOf127, {1, 2}, 0}, {{"batch_dims", 0}});
        auto Concat131 = makeOP<opset1::Concat>({Gather110, {1L}, Gather130}, {{"axis", 0}});
        auto Broadcast132 = makeOP<opset3::Broadcast>({Unsqueeze106, Concat131}, {{"mode", "bidirectional"}});
        auto Convert133 = makeOP<opset1::Convert>({Broadcast132}, {{"destination_type", "f32"}});
        auto Constant134 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {1.000000f});
        auto Multiply135 = makeOP<opset1::Multiply>({Convert133, Constant134}, {{"auto_broadcast", "numpy"}});
        auto Subtract136 = makeOP<opset1::Subtract>({Constant83, Multiply135}, {{"auto_broadcast", "numpy"}});
        auto Convert137 = makeOP<opset1::Convert>({Subtract136}, {{"destination_type", "boolean"}});
        auto Select139 = makeOP<opset1::Select>({Convert137, -FLT_MAX, Subtract136}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze140 = makeOP<opset1::Unsqueeze>({Slice126, 0});
        auto Add141 = makeOP<opset1::Add>({Select139, Unsqueeze140}, {{"auto_broadcast", "numpy"}});
        auto Multiply143 = makeOP<opset1::Multiply>({Gather116, {-1l}}, {{"auto_broadcast", "numpy"}});
        auto Slice147 = makeOP<opset8::Slice>({Add141, Multiply143, {LLONG_MAX}, {1}, {2}});
        auto sdpa =
            makeOP<v13::ScaledDotProductAttention>({Transpose62, Concat72, Concat82, Slice147}, {{"causal", false}});

        auto res = makeOP<v0::Result>({sdpa});

        ParameterVector params = nodes_to_params({beam_idx, position_ids, attention_mask, input_ids});
        model = std::make_shared<ov::Model>(OutputVector{res}, params);

        manager.register_pass<ov::pass::SDPAToPagedAttention>();
    }

    {
        auto max_context_len = make_param(PartialShape{}, element::i32, "max_context_len");
        auto block_indices_begins = make_param(PartialShape{DYN}, element::i32, "block_indices_begins");
        auto block_indices = make_param(PartialShape{DYN}, element::i32, "block_indices");
        auto subsequence_begins = make_param(PartialShape{DYN}, element::i32, "subsequence_begins");
        auto past_lens = make_param(PartialShape{DYN}, element::i32, "past_lens");
        auto value_cache_0 = make_param(PartialShape{DYN, 40, 128}, element::f32, "value_cache.0");
        auto key_cache_0 = make_param(PartialShape{DYN, 40, 128}, element::f32, "key_cache.0");
        auto input_ids = make_param(PartialShape{DYN}, element::i64, "input_ids");

        ParameterVector params = nodes_to_params({max_context_len,
                                                  block_indices_begins,
                                                  block_indices,
                                                  subsequence_begins,
                                                  past_lens,
                                                  value_cache_0,
                                                  key_cache_0,
                                                  input_ids});

        auto Constant88 = makeConst(element::u8, ov::Shape({125696, 5120}), MOCK_VALUE);
        auto Convert89 = makeOP<opset1::Convert>({Constant88}, {{"destination_type", "f16"}});
        auto Constant90 = makeConst(element::u8, ov::Shape({125696, 1}), MOCK_VALUE);
        auto Convert91 = makeOP<opset1::Convert>({Constant90}, {{"destination_type", "f16"}});
        auto Subtract92 = makeOP<opset1::Subtract>({Convert89, Convert91}, {{"auto_broadcast", "numpy"}});
        auto Constant93 = makeConst(element::f16, ov::Shape({125696, 1}), MOCK_VALUE);
        auto Multiply94 = makeOP<opset1::Multiply>({Subtract92, Constant93}, {{"auto_broadcast", "numpy"}});
        auto Convert95 = makeOP<opset1::Convert>({Multiply94}, {{"destination_type", "f32"}});
        auto Unsqueeze97 = makeOP<opset1::Unsqueeze>({input_ids, 1});
        auto Convert98 = makeOP<opset1::Convert>({Unsqueeze97}, {{"destination_type", "i32"}});
        auto Gather100 = makeOP<opset8::Gather>({Convert95, Convert98, 0}, {{"batch_dims", 0}});
        auto Constant101 = makeConst(element::f32, ov::Shape({1, 1, 5120}), MOCK_VALUE);
        auto Constant102 = makeConst(element::f32, ov::Shape({1, 1, 1}), {1.0f});
        auto Constant103 = makeConst(element::f32, ov::Shape({1, 1, 1}), {2.0f});
        auto Power104 = makeOP<opset1::Power>({Gather100, Constant103}, {{"auto_broadcast", "numpy"}});
        auto ReduceMean106 = makeOP<opset1::ReduceMean>({Power104, {-1}}, {{"keep_dims", true}});
        auto Constant107 = makeConst(element::f32, ov::Shape({1, 1, 1}), {0.000001f});
        auto Add108 = makeOP<opset1::Add>({ReduceMean106, Constant107}, {{"auto_broadcast", "numpy"}});
        auto Sqrt109 = makeOP<opset1::Sqrt>({Add108});
        auto Divide110 =
            makeOP<opset1::Divide>({Constant102, Sqrt109}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Multiply111 = makeOP<opset1::Multiply>({Gather100, Divide110}, {{"auto_broadcast", "numpy"}});
        auto Multiply112 = makeOP<opset1::Multiply>({Constant101, Multiply111}, {{"auto_broadcast", "numpy"}});
        auto Constant113 = makeConst(element::u8, ov::Shape({15360, 5120}), MOCK_VALUE);
        auto Convert114 = makeOP<opset1::Convert>({Constant113}, {{"destination_type", "f16"}});
        auto Constant115 = makeConst(element::u8, ov::Shape({15360, 1}), MOCK_VALUE);
        auto Convert116 = makeOP<opset1::Convert>({Constant115}, {{"destination_type", "f16"}});
        auto Subtract117 = makeOP<opset1::Subtract>({Convert114, Convert116}, {{"auto_broadcast", "numpy"}});
        auto Constant118 = makeConst(element::f16, ov::Shape({15360, 1}), MOCK_VALUE);
        auto Multiply119 = makeOP<opset1::Multiply>({Subtract117, Constant118}, {{"auto_broadcast", "numpy"}});
        auto Convert120 = makeOP<opset1::Convert>({Multiply119}, {{"destination_type", "f32"}});
        auto MatMul121 =
            makeOP<opset1::MatMul>({Multiply112, Convert120}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Reshape123 = makeOP<opset1::Reshape>({MatMul121, {0, 0, 3, 5120}}, {{"special_zero", true}});
        auto Unsqueeze125 = makeOP<opset1::Unsqueeze>({Reshape123, 0});
        auto Squeeze127 = makeOP<opset1::Squeeze>({Unsqueeze125, {0}});
        auto Transpose129 = makeOP<opset1::Transpose>({Squeeze127, {2, 0, 1, 3}});
        auto Gather130 = makeOP<opset8::Gather>({Transpose129, 0, 0}, {{"batch_dims", 0}});
        auto Reshape132 = makeOP<opset1::Reshape>({Gather130, {0, 0, 40, 128}}, {{"special_zero", true}});
        auto Transpose134 = makeOP<opset1::Transpose>({Reshape132, {0, 2, 1, 3}});
        auto Transpose136 = makeOP<opset1::Transpose>({Transpose134, {0, 2, 1, 3}});
        auto Reshape138 = makeOP<opset1::Reshape>({Transpose136, {0, -1}}, {{"special_zero", true}});
        auto Gather140 = makeOP<opset8::Gather>({Transpose129, 1, 0}, {{"batch_dims", 0}});
        auto Reshape142 = makeOP<opset1::Reshape>({Gather140, {0, 0, 40, 128}}, {{"special_zero", true}});
        auto Transpose144 = makeOP<opset1::Transpose>({Reshape142, {0, 2, 1, 3}});
        auto Transpose145 = makeOP<opset1::Transpose>({Transpose144, {0, 2, 1, 3}});
        auto Reshape147 = makeOP<opset1::Reshape>({Transpose145, {0, -1}}, {{"special_zero", true}});
        auto Gather149 = makeOP<opset8::Gather>({Transpose129, 2, 0}, {{"batch_dims", 0}});
        auto Reshape151 = makeOP<opset1::Reshape>({Gather149, {0, 0, 40, 128}}, {{"special_zero", true}});
        auto Transpose153 = makeOP<opset1::Transpose>({Reshape151, {0, 2, 1, 3}});
        auto Transpose154 = makeOP<opset1::Transpose>({Transpose153, {0, 2, 1, 3}});
        auto Reshape156 = makeOP<opset1::Reshape>({Transpose154, {0, -1}}, {{"special_zero", true}});
        auto Constant159 = makeConst(element::f32, ov::Shape({40, 4096, 4096}), MOCK_VALUE);
        auto Slice164 = makeOP<opset8::Slice>({Constant159, {1, 1}, {2, 2}, {1, 1}, {1, 2}});
        auto Reshape166 = makeOP<opset1::Reshape>({Slice164, {-1}}, {{"special_zero", false}});

        // PA cannot be instantiated uding makeOP hence creating constants for it manually
        auto c1 = makeConst(element::f32, {}, {0.088388f});
        auto c2 = makeConst(element::i32, {}, {0});
        auto PagedAttentionExtension168 =
            std::make_shared<ov::op::PagedAttentionExtension>(ov::OutputVector{Reshape138,
                                                                               Reshape147,
                                                                               Reshape156,
                                                                               key_cache_0,
                                                                               value_cache_0,
                                                                               past_lens,
                                                                               subsequence_begins,
                                                                               block_indices,
                                                                               block_indices_begins,
                                                                               c1,
                                                                               c2,
                                                                               Reshape166,
                                                                               max_context_len});
        auto ShapeOf172 = makeOP<opset3::ShapeOf>({Transpose154}, {{"output_type", "i64"}});
        auto Gather175 = makeOP<opset8::Gather>({ShapeOf172, -1, 0}, {{"batch_dims", 0}});
        auto Unsqueeze177 = makeOP<opset1::Unsqueeze>({Gather175, 0});
        auto Concat178 = makeOP<opset1::Concat>({{0l}, {1l}, {-1l}, Unsqueeze177}, {{"axis", 0}});
        auto Reshape179 =
            makeOP<opset1::Reshape>({PagedAttentionExtension168->output(0), Concat178}, {{"special_zero", true}});
        auto Transpose180 = makeOP<opset1::Transpose>({Reshape179, {0, 2, 1, 3}});

        auto result = std::make_shared<v0::Result>(Transpose180);
        model_ref = std::make_shared<ov::Model>(ResultVector{result}, params);

        // checks are also disabled temporarily
        comparator.disable(FunctionsComparator::PRECISIONS);
        disable_result_friendly_names_check();
        disable_rt_info_check();
    }
}

/*
As there's often a need to cover specific model's architecutres in these
tests, please, make sure you name the tests in the following manner:
SDPAToPA_MODELNAME_PATTERNYOUCOVER:
i.e. SDPAToPA_Qwen7bChat_TotalSequenceLengthPattern or
SDPAToPA_Baichuan2_13b_General if this is a test for the
entire SDPAToPA transformation
*/

const std::vector<ov::element::Type> element_types = {element::f16, element::f32};

INSTANTIATE_TEST_SUITE_P(SDPAToPATest_Conversion, SDPAToPATest, testing::ValuesIn(element_types));
