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
                                                  std::shared_ptr<Node>& head_size) {
        auto sin = makeConst(element::f32, {1, 4096, 1, 128}, MOCK_VALUE);
        auto sliced_sin_by_total = makeOP<v8::Slice>({sin, {0}, total_seq_len, {1}, {1}});
        auto rotary_emb_sin_shape = makeOP<v3::ShapeOf>({sliced_sin_by_total}, {{"output_type", "i64"}});
        head_size = makeOP<v8::Gather>({rotary_emb_sin_shape, {3}, 0}, {{"batch_dims", 0}});
        return makeOP<v8::Slice>({sliced_sin_by_total, neg_mul, {LLONG_MAX}, {1}, {1}});
    }

    static std::shared_ptr<Node> gen_rope_emb_cos(const std::shared_ptr<Node>& total_seq_len,
                                                  const std::shared_ptr<Node>& neg_mul) {
        auto cos = makeConst(element::f32, {1, 4096, 1, 128}, MOCK_VALUE);
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
                                                  std::shared_ptr<Node>& head_size) {
        auto sin = makeConst(element::f32, {1, 4096, 1, 128}, MOCK_VALUE);
        auto slice_sin = makeOP<v8::Gather>({sin, position_ids, 1}, {{"batch_dims", 0}});

        auto slice = makeOP<v8::Slice>({sin, {0}, max_context_len, {1}, {1}});
        auto shape_of = makeOP<opset3::ShapeOf>({slice}, {{"output_type", "i64"}});
        head_size = makeOP<v8::Gather>({shape_of, {3}, 0}, {{"batch_dims", 0}});

        return makeOP<v1::Reshape>({slice_sin, {-1, 1, 1, 128}}, {{"special_zero", false}});
    }

    static std::shared_ptr<Node> gen_rope_emb_cos(const std::shared_ptr<Node>& max_context_len,
                                                  const std::shared_ptr<Node>& position_ids) {
        auto cos = makeConst(element::f32, {1, 4096, 1, 128}, MOCK_VALUE);
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

TEST_F(TransformationTestsF, SDPAToPA_Qwen) {
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
        auto rope_emb_sin = Qwen7bChatSDPA::gen_rope_emb_sin(total_seq_len, neg_cur_seq_len, head_size);
        auto rope_emb_cos = Qwen7bChatSDPA::gen_rope_emb_cos(total_seq_len, neg_cur_seq_len);

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
        auto rope_emb_sin = Qwen7bChatPA::gen_rope_emb_sin(max_context_len_aligned, position_ids_aligned, head_size);
        auto rope_emb_cos = Qwen7bChatPA::gen_rope_emb_cos(max_context_len_aligned, position_ids_aligned);

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

TEST_F(TransformationTestsF, SDPAToPA_TotalSequenceLengthPatternQwen) {
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
