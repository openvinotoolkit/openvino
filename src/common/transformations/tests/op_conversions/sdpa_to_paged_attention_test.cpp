// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_not.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/einsum.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "transformations/sdpa_to_paged_attention/position_ids_replacer.hpp"
#include "transformations/sdpa_to_paged_attention/prev_sequence_length_pattern.hpp"
#include "transformations/sdpa_to_paged_attention/state_management_pattern.hpp"
#include "transformations/sdpa_to_paged_attention/total_sequence_length_pattern.hpp"
#include "transformations/utils/gen_pattern.hpp"

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

static std::shared_ptr<ov::Node> make_param(const PartialShape& pshape,
                                            element::Type element_type,
                                            const std::string& name) {
    auto param = makeOP<v0::Parameter>({}, {{"shape", pshape}, {"element_type", element_type}});
    param->set_friendly_name(name);
    param->get_output_tensor(0).set_names({name});
    return param;
}

enum QKV : int { Q = 0, K = 1, V = 2 };
vector<int> MOCK_VALUE = {1};

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
        auto beam_idx = make_param(PartialShape{DYN}, element::i64, "beam_idx");
        auto position_ids = make_param(PartialShape{DYN, DYN}, element::i64, "position_ids");
        auto attention_mask = make_param(PartialShape{DYN, DYN}, element::i64, "attention_mask");
        auto input_ids = make_param(PartialShape{DYN, DYN}, element::i64, "input_ids");
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
        auto score_aggregation_window = makeOP<v0::Parameter>({}, {{"shape", PartialShape{DYN}}, el_type_i32});

        auto rotated_block_indices = makeConst(element::i32, ov::Shape({0}), {0});
        auto rotation_deltas = makeConst(element::i32, ov::Shape{0}, {0});
        auto rotation_trig_lut = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_threshold = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_block_size = makeConst(element::i32, ov::Shape({}), MOCK_VALUE);
        auto xattention_stride = makeConst(element::i32, ov::Shape({}), MOCK_VALUE);

        auto params = nodes_to_params({score_aggregation_window,
                                       max_context_len,
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
        auto score_aggregation_window_const = std::make_shared<v0::Constant>(element::i32, Shape{0}, 0);
        auto sinks = v0::Constant::create(element::f32, Shape{0, 0, 0, 0}, {});

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
                                                                             max_context_len,
                                                                             score_aggregation_window_const,
                                                                             rotated_block_indices,
                                                                             rotation_deltas,
                                                                             rotation_trig_lut,
                                                                             xattention_threshold,
                                                                             xattention_block_size,
                                                                             xattention_stride,
                                                                             sinks});
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

TEST_F(SDPAToPATest, SDPAToPA_Qwen7bChat_TotalSequenceLengthPattern) {
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

TEST_F(SDPAToPATest, SDPAToPA_Qwen7bChat_PositionIDsReplacerQwenPattern) {
    {
        auto max_context_len = std::make_shared<v0::Parameter>(element::i32, PartialShape{});
        auto max_context_len_i64 = std::make_shared<v0::Convert>(max_context_len, element::i64);
        auto max_context_len_reshaped =
            std::make_shared<v1::Reshape>(max_context_len_i64, v0::Constant::create(element::i64, Shape{1}, {1}), true);
        max_context_len->set_friendly_name("max_context_len");

        auto rotary_emb_sincos = std::make_shared<v0::Parameter>(element::f32, PartialShape{1, DYN, 1, 128});
        auto position_ids = std::make_shared<v0::Parameter>(element::i64, PartialShape{DYN});

        auto fake_input = std::make_shared<v0::Parameter>(element::i64, PartialShape{DYN, DYN});
        auto shape = std::make_shared<v3::ShapeOf>(fake_input, element::i64);
        auto gather = std::make_shared<v8::Gather>(shape,
                                                   v0::Constant::create(element::i64, Shape{1}, {1}),
                                                   v0::Constant::create(element::i64, Shape{1}, {0}));

        auto minus_one = v0::Constant::create(element::i32, Shape{1}, {-1});
        auto minus_one_converted = std::make_shared<v0::Convert>(minus_one, element::i64);
        auto minus_one_reshaped = std::make_shared<v1::Reshape>(minus_one_converted,
                                                                v0::Constant::create(element::i64, Shape{1}, {-1}),
                                                                true);
        auto past_offset = std::make_shared<v1::Multiply>(gather, minus_one_reshaped);

        auto start_const = v0::Constant::create(element::i64, Shape{1}, {0});
        auto stop_const = v0::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>().max()});
        auto step_const = v0::Constant::create(element::i64, Shape{1}, {1});
        auto axis_const = v0::Constant::create(element::i64, Shape{1}, {1});

        auto slice_1 = std::make_shared<v8::Slice>(rotary_emb_sincos,
                                                   start_const,
                                                   max_context_len_reshaped,
                                                   step_const,
                                                   axis_const);
        auto slice_2 = std::make_shared<v8::Slice>(slice_1, past_offset, stop_const, step_const, axis_const);
        auto result = std::make_shared<v0::Result>(slice_2);

        model = std::make_shared<Model>(ResultVector{result},
                                        ParameterVector{max_context_len, rotary_emb_sincos, fake_input, position_ids});
        manager.register_pass<pass::PositionIDsReplacerQwen>(position_ids);
    }

    {
        auto rotary_emb_sincos = std::make_shared<v0::Parameter>(element::f32, PartialShape{1, DYN, 1, 128});
        auto position_ids = std::make_shared<v0::Parameter>(element::i64, PartialShape{DYN});

        auto gather_new = std::make_shared<v8::Gather>(rotary_emb_sincos,
                                                       position_ids,
                                                       v0::Constant::create(element::i64, Shape{}, {1}));
        auto new_shape = v0::Constant::create(element::i64, Shape{4}, {-1, 1, 1, 128});
        auto reshaped = std::make_shared<v1::Reshape>(gather_new, new_shape, true);

        model_ref = std::make_shared<Model>(OutputVector{reshaped}, ParameterVector{rotary_emb_sincos, position_ids});
    }

    // TODO: align precisions, check the copying of "fuse_names" attr in SDPAToPagedAttention
    // checking the graph structure and names, other checks are temporarily disabled:
    comparator.disable(FunctionsComparator::PRECISIONS);
    disable_result_friendly_names_check();
    disable_rt_info_check();
}

// TODO: split the models in blocks the way it's done for Qwen and make the code not to be such a clutter
// TODO: write a test for StateManagementPattern only (because changes for Alibi are inside it)
// TODO: align precisions, check the copying of "fuse_names" attr in SDPAToPagedAttention
// checking the graph structure and names, other checks are temporarily disabled:
TEST_F(SDPAToPATest, SDPAToPA_Baichuan2_13b_General) {
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
        auto score_aggregation_window = makeConst(element::i32, ov::Shape({0}), MOCK_VALUE);
        auto rotated_block_indices = makeConst(element::i32, ov::Shape({0}), {0});
        auto rotation_deltas = makeConst(element::i32, ov::Shape{0}, {0});
        auto rotation_trig_lut = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_threshold = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_block_size = makeConst(element::i32, ov::Shape({}), MOCK_VALUE);
        auto xattention_stride = makeConst(element::i32, ov::Shape({}), MOCK_VALUE);

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
        auto sinks = v0::Constant::create(element::f32, Shape{0, 0, 0, 0}, {});
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
                                                                               max_context_len,
                                                                               score_aggregation_window,
                                                                               rotated_block_indices,
                                                                               rotation_deltas,
                                                                               rotation_trig_lut,
                                                                               xattention_threshold,
                                                                               xattention_block_size,
                                                                               xattention_stride,
                                                                               sinks});
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

// todo: split the code to functional blocks as for Qwen-7b model
TEST_F(SDPAToPATest, SDPAToPA_nanoLLaVA_General) {
    {
        auto beam_idx = make_param(PartialShape{DYN}, element::i32, "beam_idx");
        auto inputs_embeds = make_param(PartialShape{DYN, DYN, 8}, element::f32, "inputs_embeds");
        auto position_ids = make_param(PartialShape{DYN, DYN}, element::i64, "position_ids");
        auto attention_mask = make_param(PartialShape{DYN, DYN}, element::i64, "attention_mask");

        auto ShapeOf_19592 = makeOP<opset3::ShapeOf>({inputs_embeds}, {{"output_type", "i64"}});
        auto Gather_19597 = makeOP<opset8::Gather>({ShapeOf_19592, {0}, 0}, {{"batch_dims", 0}});
        auto Concat_19604 = makeOP<opset1::Concat>({Gather_19597, {2l}, {0l}, {2l}}, {{"axis", 0}});
        auto Broadcast_19607 = makeOP<opset3::Broadcast>({0.000000f, Concat_19604}, {{"mode", "numpy"}});
        auto ReadValue_19126 = makeOP<opset6::ReadValue>(
            {Broadcast_19607},
            {{"variable_id", "var1"}, {"variable_type", "f32"}, {"variable_shape", PartialShape{DYN, 2, DYN, 2}}});
        auto Gather_18655 = makeOP<opset8::Gather>({ReadValue_19126, beam_idx, 0}, {{"batch_dims", 0}});
        auto Constant_16156 =
            makeConst(element::f32,
                      ov::Shape({1, 1, 8}),
                      {1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f});
        auto Constant_16155 = makeConst(element::f32, ov::Shape({1, 1, 1}), {1.000000f});
        auto Constant_16153 = makeConst(element::f32, ov::Shape({1, 1, 1}), {2.000000f});
        auto pref_7_pow_Power = makeOP<opset1::Power>({inputs_embeds, Constant_16153}, {{"auto_broadcast", "numpy"}});
        auto pref_7_mean_ReduceMean = makeOP<opset1::ReduceMean>({pref_7_pow_Power, {-1}}, {{"keep_dims", true}});
        auto Constant_16154 = makeConst(element::f32, ov::Shape({1, 1, 1}), {0.000001f});
        auto pref_7_add_Add =
            makeOP<opset1::Add>({pref_7_mean_ReduceMean, Constant_16154}, {{"auto_broadcast", "numpy"}});
        auto pref_7_rsqrt_Sqrt = makeOP<opset1::Sqrt>({pref_7_add_Add});
        auto pref_7_rsqrt_Divide = makeOP<opset1::Divide>({Constant_16155, pref_7_rsqrt_Sqrt},
                                                          {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto pref_7_mul_Multiply =
            makeOP<opset1::Multiply>({inputs_embeds, pref_7_rsqrt_Divide}, {{"auto_broadcast", "numpy"}});
        auto pref_7_mul_Multiply_1 =
            makeOP<opset1::Multiply>({Constant_16156, pref_7_mul_Multiply}, {{"auto_broadcast", "numpy"}});
        auto self_model_model_layers_0_self_attn_q_proj_weight = makeConst(element::f32, ov::Shape({8, 8}), MOCK_VALUE);
        auto __module_model_model_layers_0_self_attn_q_proj_aten_linear_MatMul =
            makeOP<opset1::MatMul>({pref_7_mul_Multiply_1, self_model_model_layers_0_self_attn_q_proj_weight},
                                   {{"transpose_a", false}, {"transpose_b", true}});
        auto pref_1_view_Reshape =
            makeOP<opset1::Reshape>({__module_model_model_layers_0_self_attn_q_proj_aten_linear_MatMul, {0, 0, 4, 2}},
                                    {{"special_zero", true}});
        auto pref_1_transpose_Transpose = makeOP<opset1::Transpose>({pref_1_view_Reshape, {0, 2, 1, 3}});
        auto self_model_model_layers_0_self_attn_rotary_emb_cos_cached =
            makeConst(element::f32, ov::Shape({32768, 2}), MOCK_VALUE);
        auto ShapeOf_16753 = makeOP<opset3::ShapeOf>({pref_7_mul_Multiply_1}, {{"output_type", "i64"}});
        auto Gather_16756 = makeOP<opset8::Gather>({ShapeOf_16753, 1, 0}, {{"batch_dims", 0}});
        auto Reshape_16764 = makeOP<opset1::Reshape>({Gather_16756, {-1}}, {{"special_zero", false}});
        auto ReadValue_19120 = makeOP<opset6::ReadValue>(
            {Broadcast_19607},
            {{"variable_id", "var2"}, {"variable_type", "f32"}, {"variable_shape", PartialShape{DYN, 2, DYN, 2}}});
        auto Gather_18646 = makeOP<opset8::Gather>({ReadValue_19120, beam_idx, 0}, {{"batch_dims", 0}});
        auto ShapeOf_16767 = makeOP<opset3::ShapeOf>({Gather_18646}, {{"output_type", "i64"}});
        auto Gather_16770 = makeOP<opset8::Gather>({ShapeOf_16767, 2, 0}, {{"batch_dims", 0}});
        auto Reshape_16772 = makeOP<opset1::Reshape>({Gather_16770, {-1}}, {{"special_zero", false}});
        auto pref_1_add__Add = makeOP<opset1::Add>({Reshape_16764, Reshape_16772}, {{"auto_broadcast", "numpy"}});
        auto pref_2_slice_Slice = makeOP<opset8::Slice>(
            {self_model_model_layers_0_self_attn_rotary_emb_cos_cached, {0}, pref_1_add__Add, {1}, {0}});
        auto pref_6_view_Reshape = makeOP<opset1::Reshape>({position_ids, {0, 0}}, {{"special_zero", true}});
        auto pref_1_index_Convert = makeOP<opset1::Convert>({pref_6_view_Reshape}, {{"destination_type", "i32"}});
        auto pref_1_index_Gather =
            makeOP<opset8::Gather>({pref_2_slice_Slice, pref_1_index_Convert, 0}, {{"batch_dims", 0}});
        auto pref_1_unsqueeze_Unsqueeze = makeOP<opset1::Unsqueeze>({pref_1_index_Gather, 1});
        auto pref_1_mul_Multiply = makeOP<opset1::Multiply>({pref_1_transpose_Transpose, pref_1_unsqueeze_Unsqueeze},
                                                            {{"auto_broadcast", "numpy"}});
        auto pref_1_slice_Slice = makeOP<opset8::Slice>({pref_1_transpose_Transpose, {1}, {LLONG_MAX}, {1}, {3}});
        auto Constant_16157 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {-1.000000f});
        auto pref_1_neg_Multiply =
            makeOP<opset1::Multiply>({pref_1_slice_Slice, Constant_16157}, {{"auto_broadcast", "numpy"}});
        auto pref_1_slice_Slice_1 = makeOP<opset8::Slice>({pref_1_transpose_Transpose, {0}, {1}, {1}, {3}});
        auto pref_1_cat_Concat = makeOP<opset1::Concat>({pref_1_neg_Multiply, pref_1_slice_Slice_1}, {{"axis", -1}});
        auto self_model_model_layers_0_self_attn_rotary_emb_sin_cached =
            makeConst(element::f32, ov::Shape({32768, 2}), MOCK_VALUE);
        auto pref_2_slice_Slice_1 = makeOP<opset8::Slice>(
            {self_model_model_layers_0_self_attn_rotary_emb_sin_cached, {0}, pref_1_add__Add, {1}, {0}});
        auto pref_1_index_Gather_1 =
            makeOP<opset8::Gather>({pref_2_slice_Slice_1, pref_1_index_Convert, 0}, {{"batch_dims", 0}});
        auto pref_1_unsqueeze_Unsqueeze_1 = makeOP<opset1::Unsqueeze>({pref_1_index_Gather_1, 1});
        auto pref_1_mul_Multiply_1 =
            makeOP<opset1::Multiply>({pref_1_cat_Concat, pref_1_unsqueeze_Unsqueeze_1}, {{"auto_broadcast", "numpy"}});
        auto pref_1_add_Add =
            makeOP<opset1::Add>({pref_1_mul_Multiply, pref_1_mul_Multiply_1}, {{"auto_broadcast", "numpy"}});
        auto pref_8_weight = makeConst(element::f32, ov::Shape({4, 8}), MOCK_VALUE);
        auto pref_3_linear_MatMul = makeOP<opset1::MatMul>({pref_7_mul_Multiply_1, pref_8_weight},
                                                           {{"transpose_a", false}, {"transpose_b", true}});
        auto pref_1_view_Reshape_1 =
            makeOP<opset1::Reshape>({pref_3_linear_MatMul, {0, 0, 2, 2}}, {{"special_zero", true}});
        auto pref_1_transpose_Transpose_1 = makeOP<opset1::Transpose>({pref_1_view_Reshape_1, {0, 2, 1, 3}});
        auto pref_1_mul_Multiply_2 =
            makeOP<opset1::Multiply>({pref_1_transpose_Transpose_1, pref_1_unsqueeze_Unsqueeze},
                                     {{"auto_broadcast", "numpy"}});
        auto pref_1_slice_Slice_2 = makeOP<opset8::Slice>({pref_1_transpose_Transpose_1, {1}, {LLONG_MAX}, {1}, {3}});
        auto Constant_16158 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {-1.000000f});
        auto pref_1_neg_Multiply_1 =
            makeOP<opset1::Multiply>({pref_1_slice_Slice_2, Constant_16158}, {{"auto_broadcast", "numpy"}});
        auto pref_1_slice_Slice_3 = makeOP<opset8::Slice>({pref_1_transpose_Transpose_1, {0}, {1}, {1}, {3}});
        auto pref_1_cat_Concat_1 =
            makeOP<opset1::Concat>({pref_1_neg_Multiply_1, pref_1_slice_Slice_3}, {{"axis", -1}});
        auto pref_1_mul_Multiply_3 = makeOP<opset1::Multiply>({pref_1_cat_Concat_1, pref_1_unsqueeze_Unsqueeze_1},
                                                              {{"auto_broadcast", "numpy"}});
        auto pref_1_add_Add_1 =
            makeOP<opset1::Add>({pref_1_mul_Multiply_2, pref_1_mul_Multiply_3}, {{"auto_broadcast", "numpy"}});
        auto pref_1_cat_Concat_2 = makeOP<opset1::Concat>({Gather_18646, pref_1_add_Add_1}, {{"axis", -2}});
        auto pref_1_unsqueeze_Unsqueeze_2 = makeOP<opset1::Unsqueeze>({pref_1_cat_Concat_2, 2});
        auto Gather_16778 = makeOP<opset8::Gather>({ShapeOf_16753, {0}, 0}, {{"batch_dims", 0}});
        auto Add_16793 = makeOP<opset1::Add>({Reshape_16772, Reshape_16764}, {{"auto_broadcast", "numpy"}});
        auto pref_4_ListConstruct_2 =
            makeOP<opset1::Concat>({Gather_16778, {2l}, {2l}, Add_16793, {2l}}, {{"axis", 0}});
        auto pref_1_expand_Broadcast = makeOP<opset3::Broadcast>({pref_1_unsqueeze_Unsqueeze_2, pref_4_ListConstruct_2},
                                                                 {{"mode", "bidirectional"}});
        auto pref_1_reshape_Reshape =
            makeOP<opset1::Reshape>({pref_1_expand_Broadcast, {0, 4, -1, 2}}, {{"special_zero", true}});
        auto ReadValue_19122 = makeOP<opset6::ReadValue>(
            {Broadcast_19607},
            {{"variable_id", "var3"}, {"variable_type", "f32"}, {"variable_shape", PartialShape{DYN, 2, DYN, 2}}});
        auto Gather_18649 = makeOP<opset8::Gather>({ReadValue_19122, beam_idx, 0}, {{"batch_dims", 0}});
        auto self_model_model_layers_0_self_attn_v_proj_weight = makeConst(element::f32, ov::Shape({4, 8}), MOCK_VALUE);
        auto pref_9_MatMul =
            makeOP<opset1::MatMul>({pref_7_mul_Multiply_1, self_model_model_layers_0_self_attn_v_proj_weight},
                                   {{"transpose_a", false}, {"transpose_b", true}});
        auto pref_1_view_Reshape_2 = makeOP<opset1::Reshape>({pref_9_MatMul, {0, 0, 2, 2}}, {{"special_zero", true}});
        auto pref_1_transpose_Transpose_2 = makeOP<opset1::Transpose>({pref_1_view_Reshape_2, {0, 2, 1, 3}});
        auto pref_1_cat_Concat_3 = makeOP<opset1::Concat>({Gather_18649, pref_1_transpose_Transpose_2}, {{"axis", -2}});
        auto pref_1_unsqueeze_Unsqueeze_3 = makeOP<opset1::Unsqueeze>({pref_1_cat_Concat_3, 2});
        auto pref_1_expand_Broadcast_1 =
            makeOP<opset3::Broadcast>({pref_1_unsqueeze_Unsqueeze_3, pref_4_ListConstruct_2},
                                      {{"mode", "bidirectional"}});
        auto pref_1_reshape_Reshape_1 =
            makeOP<opset1::Reshape>({pref_1_expand_Broadcast_1, {0, 4, -1, 2}}, {{"special_zero", true}});
        auto Constant_16160 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {1.000000f});
        auto pref_6_unsqueeze_Unsqueeze = makeOP<opset1::Unsqueeze>({attention_mask, 1});
        auto pref_6_unsqueeze_Unsqueeze_1 = makeOP<opset1::Unsqueeze>({pref_6_unsqueeze_Unsqueeze, 2});
        auto ShapeOf_16779 = makeOP<opset3::ShapeOf>({attention_mask}, {{"output_type", "i64"}});
        auto Gather_16782 = makeOP<opset8::Gather>({ShapeOf_16779, {1}, 0}, {{"batch_dims", 0}});
        auto pref_5_ListConstruct_1 =
            makeOP<opset1::Concat>({Gather_16778, {1l}, Reshape_16764, Gather_16782}, {{"axis", 0}});
        auto pref_6_expand_Broadcast = makeOP<opset3::Broadcast>({pref_6_unsqueeze_Unsqueeze_1, pref_5_ListConstruct_1},
                                                                 {{"mode", "bidirectional"}});
        auto pref_6_to_Convert_1 = makeOP<opset1::Convert>({pref_6_expand_Broadcast}, {{"destination_type", "f32"}});
        auto Constant_16159 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {1.000000f});
        auto pref_6_rsub_Multiply =
            makeOP<opset1::Multiply>({pref_6_to_Convert_1, Constant_16159}, {{"auto_broadcast", "numpy"}});
        auto pref_6_rsub_Subtract =
            makeOP<opset1::Subtract>({Constant_16160, pref_6_rsub_Multiply}, {{"auto_broadcast", "numpy"}});
        auto pref_6_to_Convert_2 = makeOP<opset1::Convert>({pref_6_rsub_Subtract}, {{"destination_type", "boolean"}});
        auto pref_6_masked_fill_Select = makeOP<opset1::Select>({pref_6_to_Convert_2, -FLT_MAX, pref_6_rsub_Subtract},
                                                                {{"auto_broadcast", "numpy"}});
        auto pref_6_to_Convert_4 =
            makeOP<opset1::Convert>({pref_6_masked_fill_Select}, {{"destination_type", "boolean"}});
        auto pref_6_add_Add = makeOP<opset1::Add>({Gather_16756, Gather_16770}, {{"auto_broadcast", "numpy"}});
        auto pref_6_sub_Subtract =
            makeOP<opset1::Subtract>({pref_6_add_Add, Gather_16756}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_124 = makeOP<opset1::Unsqueeze>({pref_6_sub_Subtract, 0});
        auto pref_5_ListConstruct_2 = makeOP<opset1::Concat>({Reshape_16764, Unsqueeze_124}, {{"axis", 0}});
        auto pref_6_zeros_Broadcast =
            makeOP<opset3::Broadcast>({0.000000f, pref_5_ListConstruct_2}, {{"mode", "numpy"}});
        auto pref_6_arange_Range = makeOP<opset4::Range>({0, Gather_16756, 1}, {{"output_type", "f32"}});
        auto pref_6_arange_ConvertLike = makeOP<opset1::Convert>({pref_6_arange_Range}, {{"destination_type", "i64"}});
        auto pref_6_add_Add_1 = makeOP<opset1::Add>({pref_6_arange_ConvertLike, {1l}}, {{"auto_broadcast", "numpy"}});
        auto pref_6_view_Reshape_1 = makeOP<opset1::Reshape>({pref_6_add_Add_1, {0, 1}}, {{"special_zero", true}});
        auto pref_6_lt_Less =
            makeOP<opset1::Less>({pref_6_arange_ConvertLike, pref_6_view_Reshape_1}, {{"auto_broadcast", "numpy"}});
        auto pref_5_ListConstruct_3 = makeOP<opset3::Broadcast>({Reshape_16764, {2}}, {{"mode", "numpy"}});
        auto pref_6_full_Broadcast = makeOP<opset3::Broadcast>({-FLT_MAX, pref_5_ListConstruct_3}, {{"mode", "numpy"}});
        auto pref_6_masked_fill__Select =
            makeOP<opset1::Select>({pref_6_lt_Less, 0.000000f, pref_6_full_Broadcast}, {{"auto_broadcast", "numpy"}});
        auto pref_6_cat_Concat =
            makeOP<opset1::Concat>({pref_6_zeros_Broadcast, pref_6_masked_fill__Select}, {{"axis", -1}});
        auto pref_6_unsqueeze_Unsqueeze_2 = makeOP<opset1::Unsqueeze>({pref_6_cat_Concat, 0});
        auto pref_6_unsqueeze_Unsqueeze_3 = makeOP<opset1::Unsqueeze>({pref_6_unsqueeze_Unsqueeze_2, 1});
        auto pref_6_add_Add_2 = makeOP<opset1::Add>({Reshape_16764, Unsqueeze_124}, {{"auto_broadcast", "numpy"}});
        auto pref_5_ListConstruct_5 =
            makeOP<opset1::Concat>({Gather_16778, {1l}, Reshape_16764, pref_6_add_Add_2}, {{"axis", 0}});
        auto pref_6_expand_Broadcast_1 =
            makeOP<opset3::Broadcast>({pref_6_unsqueeze_Unsqueeze_3, pref_5_ListConstruct_5},
                                      {{"mode", "bidirectional"}});
        auto pref_6_masked_fill_Select_1 =
            makeOP<opset1::Select>({pref_6_to_Convert_4, -FLT_MAX, pref_6_expand_Broadcast_1},
                                   {{"auto_broadcast", "numpy"}});
        auto sdpa = makeOP<v13::ScaledDotProductAttention>(
            {pref_1_add_Add, pref_1_reshape_Reshape, pref_1_reshape_Reshape_1, pref_6_masked_fill_Select_1},
            {{"causal", false}});

        auto res = makeOP<v0::Result>({sdpa});

        ParameterVector params = nodes_to_params({beam_idx, position_ids, attention_mask, inputs_embeds});
        model = std::make_shared<ov::Model>(OutputVector{res}, params);

        manager.register_pass<ov::pass::SDPAToPagedAttention>();
    }

    {
        auto max_context_len = make_param(PartialShape{}, element::i32, "max_context_len");
        auto block_indices_begins = make_param(PartialShape{DYN}, element::i32, "block_indices_begins");
        auto block_indices = make_param(PartialShape{DYN}, element::i32, "block_indices");
        auto subsequence_begins = make_param(PartialShape{DYN}, element::i32, "subsequence_begins");
        auto past_lens = make_param(PartialShape{DYN}, element::i32, "past_lens");
        auto value_cache_0 = make_param(PartialShape{DYN, 2, 2}, element::f32, "value_cache_0");
        auto key_cache_0 = make_param(PartialShape{DYN, 2, 2}, element::f32, "key_cache_0");
        auto inputs_embeds = make_param(PartialShape{DYN, DYN}, element::f32, "inputs_embeds");
        auto position_ids = make_param(PartialShape{DYN}, element::i64, "position_ids");
        auto score_aggregation_window = makeConst(element::i32, ov::Shape({0}), MOCK_VALUE);
        auto rotated_block_indices = makeConst(element::i32, ov::Shape({0}), {0});
        auto rotation_deltas = makeConst(element::i32, ov::Shape{0}, {0});
        auto rotation_trig_lut = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_threshold = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_block_size = makeConst(element::i32, ov::Shape({}), MOCK_VALUE);
        auto xattention_stride = makeConst(element::i32, ov::Shape({}), MOCK_VALUE);

        ParameterVector params = nodes_to_params({max_context_len,
                                                  block_indices_begins,
                                                  block_indices,
                                                  subsequence_begins,
                                                  past_lens,
                                                  value_cache_0,
                                                  key_cache_0,
                                                  inputs_embeds,
                                                  position_ids});

        auto Constant_16156 =
            makeConst(element::f32,
                      ov::Shape({1, 1, 8}),
                      {1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f});
        auto Constant_16155 = makeConst(element::f32, ov::Shape({1, 1, 1}), {1.000000f});
        auto Constant_16153 = makeConst(element::f32, ov::Shape({1, 1, 1}), {2.000000f});
        auto unsqueezed_inputs_embeds = makeOP<opset1::Unsqueeze>({inputs_embeds, 1});
        auto pref_7_pow_Power =
            makeOP<opset1::Power>({unsqueezed_inputs_embeds, Constant_16153}, {{"auto_broadcast", "numpy"}});
        auto pref_7_mean_ReduceMean = makeOP<opset1::ReduceMean>({pref_7_pow_Power, {-1}}, {{"keep_dims", true}});
        auto Constant_16154 = makeConst(element::f32, ov::Shape({1, 1, 1}), {0.000001f});
        auto pref_7_add_Add =
            makeOP<opset1::Add>({pref_7_mean_ReduceMean, Constant_16154}, {{"auto_broadcast", "numpy"}});
        auto pref_7_rsqrt_Sqrt = makeOP<opset1::Sqrt>({pref_7_add_Add});
        auto pref_7_rsqrt_Divide = makeOP<opset1::Divide>({Constant_16155, pref_7_rsqrt_Sqrt},
                                                          {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto pref_7_mul_Multiply =
            makeOP<opset1::Multiply>({unsqueezed_inputs_embeds, pref_7_rsqrt_Divide}, {{"auto_broadcast", "numpy"}});
        auto pref_7_mul_Multiply_1 =
            makeOP<opset1::Multiply>({Constant_16156, pref_7_mul_Multiply}, {{"auto_broadcast", "numpy"}});
        auto self_model_model_layers_0_self_attn_q_proj_weight = makeConst(element::f32, ov::Shape({8, 8}), MOCK_VALUE);
        auto __module_model_model_layers_0_self_attn_q_proj_aten_linear_MatMul =
            makeOP<opset1::MatMul>({pref_7_mul_Multiply_1, self_model_model_layers_0_self_attn_q_proj_weight},
                                   {{"transpose_a", false}, {"transpose_b", true}});
        auto pref_1_view_Reshape =
            makeOP<opset1::Reshape>({__module_model_model_layers_0_self_attn_q_proj_aten_linear_MatMul, {0, 0, 4, 2}},
                                    {{"special_zero", true}});
        auto pref_1_transpose_Transpose = makeOP<opset1::Transpose>({pref_1_view_Reshape, {0, 2, 1, 3}});
        auto self_model_model_layers_0_self_attn_rotary_emb_cos_cached =
            makeConst(element::f32, ov::Shape({32768, 2}), MOCK_VALUE);
        auto ShapeOf_16753 = makeOP<opset3::ShapeOf>({pref_7_mul_Multiply_1}, {{"output_type", "i64"}});
        auto Gather_16756 = makeOP<opset8::Gather>({ShapeOf_16753, 1, 0}, {{"batch_dims", 0}});
        auto Reshape_16764 = makeOP<opset1::Reshape>({Gather_16756, {-1}}, {{"special_zero", false}});
        auto ShapeOf_52004 = makeOP<opset3::ShapeOf>({unsqueezed_inputs_embeds}, {{"output_type", "i64"}});
        auto Gather_52005 = makeOP<opset8::Gather>({ShapeOf_52004, 1, 0}, {{"batch_dims", 0}});
        auto Convert_52006 = makeOP<opset1::Convert>({Gather_52005}, {{"destination_type", "i32"}});
        auto Subtract_52007 = makeOP<opset1::Subtract>({max_context_len, Convert_52006}, {{"auto_broadcast", "numpy"}});
        auto Convert_52008 = makeOP<opset1::Convert>({Subtract_52007}, {{"destination_type", "i64"}});
        auto Reshape_16772 = makeOP<opset1::Reshape>({Convert_52008, {-1}}, {{"special_zero", false}});
        auto pref_1_add__Add = makeOP<opset1::Add>({Reshape_16764, Reshape_16772}, {{"auto_broadcast", "numpy"}});
        auto pref_2_slice_Slice = makeOP<opset8::Slice>(
            {self_model_model_layers_0_self_attn_rotary_emb_cos_cached, {0}, pref_1_add__Add, {1}, {0}});
        auto Unsqueeze_51575 = makeOP<opset1::Unsqueeze>({position_ids, 1});
        auto pref_6_view_Reshape = makeOP<opset1::Reshape>({Unsqueeze_51575, {0, 0}}, {{"special_zero", true}});
        auto pref_1_index_Convert = makeOP<opset1::Convert>({pref_6_view_Reshape}, {{"destination_type", "i32"}});
        auto pref_1_index_Gather =
            makeOP<opset8::Gather>({pref_2_slice_Slice, pref_1_index_Convert, 0}, {{"batch_dims", 0}});
        auto pref_1_unsqueeze_Unsqueeze = makeOP<opset1::Unsqueeze>({pref_1_index_Gather, 1});
        auto pref_1_mul_Multiply = makeOP<opset1::Multiply>({pref_1_transpose_Transpose, pref_1_unsqueeze_Unsqueeze},
                                                            {{"auto_broadcast", "numpy"}});
        auto pref_1_slice_Slice = makeOP<opset8::Slice>({pref_1_transpose_Transpose, {1}, {LLONG_MAX}, {1}, {3}});
        auto Constant_16157 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {-1.000000f});
        auto pref_1_neg_Multiply =
            makeOP<opset1::Multiply>({pref_1_slice_Slice, Constant_16157}, {{"auto_broadcast", "numpy"}});
        auto pref_1_slice_Slice_1 = makeOP<opset8::Slice>({pref_1_transpose_Transpose, {0}, {1}, {1}, {3}});
        auto pref_1_cat_Concat = makeOP<opset1::Concat>({pref_1_neg_Multiply, pref_1_slice_Slice_1}, {{"axis", -1}});
        auto self_model_model_layers_0_self_attn_rotary_emb_sin_cached =
            makeConst(element::f32, ov::Shape({32768, 2}), MOCK_VALUE);
        auto pref_2_slice_Slice_1 = makeOP<opset8::Slice>(
            {self_model_model_layers_0_self_attn_rotary_emb_sin_cached, {0}, pref_1_add__Add, {1}, {0}});
        auto pref_1_index_Gather_1 =
            makeOP<opset8::Gather>({pref_2_slice_Slice_1, pref_1_index_Convert, 0}, {{"batch_dims", 0}});
        auto pref_1_unsqueeze_Unsqueeze_1 = makeOP<opset1::Unsqueeze>({pref_1_index_Gather_1, 1});
        auto pref_1_mul_Multiply_1 =
            makeOP<opset1::Multiply>({pref_1_cat_Concat, pref_1_unsqueeze_Unsqueeze_1}, {{"auto_broadcast", "numpy"}});
        auto pref_1_add_Add =
            makeOP<opset1::Add>({pref_1_mul_Multiply, pref_1_mul_Multiply_1}, {{"auto_broadcast", "numpy"}});
        auto Transpose_51951 = makeOP<opset1::Transpose>({pref_1_add_Add, {0, 2, 1, 3}});
        auto Reshape_51953 = makeOP<opset1::Reshape>({Transpose_51951, {0, -1}}, {{"special_zero", true}});
        auto pref_8_weight = makeConst(element::f32, ov::Shape({4, 8}), MOCK_VALUE);
        auto pref_3_linear_MatMul = makeOP<opset1::MatMul>({pref_7_mul_Multiply_1, pref_8_weight},
                                                           {{"transpose_a", false}, {"transpose_b", true}});
        auto pref_1_view_Reshape_1 =
            makeOP<opset1::Reshape>({pref_3_linear_MatMul, {0, 0, 2, 2}}, {{"special_zero", true}});
        auto pref_1_transpose_Transpose_1 = makeOP<opset1::Transpose>({pref_1_view_Reshape_1, {0, 2, 1, 3}});
        auto pref_1_mul_Multiply_2 =
            makeOP<opset1::Multiply>({pref_1_transpose_Transpose_1, pref_1_unsqueeze_Unsqueeze},
                                     {{"auto_broadcast", "numpy"}});
        auto pref_1_slice_Slice_2 = makeOP<opset8::Slice>({pref_1_transpose_Transpose_1, {1}, {LLONG_MAX}, {1}, {3}});
        auto Constant_16158 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {-1.000000f});
        auto pref_1_neg_Multiply_1 =
            makeOP<opset1::Multiply>({pref_1_slice_Slice_2, Constant_16158}, {{"auto_broadcast", "numpy"}});
        auto pref_1_slice_Slice_3 = makeOP<opset8::Slice>({pref_1_transpose_Transpose_1, {0}, {1}, {1}, {3}});
        auto pref_1_cat_Concat_1 =
            makeOP<opset1::Concat>({pref_1_neg_Multiply_1, pref_1_slice_Slice_3}, {{"axis", -1}});
        auto pref_1_mul_Multiply_3 = makeOP<opset1::Multiply>({pref_1_cat_Concat_1, pref_1_unsqueeze_Unsqueeze_1},
                                                              {{"auto_broadcast", "numpy"}});
        auto pref_1_add_Add_1 =
            makeOP<opset1::Add>({pref_1_mul_Multiply_2, pref_1_mul_Multiply_3}, {{"auto_broadcast", "numpy"}});
        auto Transpose_51954 = makeOP<opset1::Transpose>({pref_1_add_Add_1, {0, 2, 1, 3}});
        auto Reshape_51957 = makeOP<opset1::Reshape>({Transpose_51954, {0, -1}}, {{"special_zero", true}});
        auto self_model_model_layers_0_self_attn_v_proj_weight = makeConst(element::f32, ov::Shape({4, 8}), MOCK_VALUE);
        auto pref_9_MatMul =
            makeOP<opset1::MatMul>({pref_7_mul_Multiply_1, self_model_model_layers_0_self_attn_v_proj_weight},
                                   {{"transpose_a", false}, {"transpose_b", true}});
        auto pref_1_view_Reshape_2 = makeOP<opset1::Reshape>({pref_9_MatMul, {0, 0, 2, 2}}, {{"special_zero", true}});
        auto pref_1_transpose_Transpose_2 = makeOP<opset1::Transpose>({pref_1_view_Reshape_2, {0, 2, 1, 3}});
        auto Transpose_51955 = makeOP<opset1::Transpose>({pref_1_transpose_Transpose_2, {0, 2, 1, 3}});
        auto Reshape_51959 = makeOP<opset1::Reshape>({Transpose_51955, {0, -1}}, {{"special_zero", true}});

        auto c1 = makeConst(element::f32, {}, {0.707107f});
        auto c2 = makeConst(element::i32, {}, {0});
        // an empty Constant needs to be created in a usual way, not using makeConst()
        auto c3 = v0::Constant::create(element::f32, {0}, {});
        auto sinks = v0::Constant::create(element::f32, Shape{0, 0, 0, 0}, {});
        auto PagedAttentionExtension_51962 =
            std::make_shared<ov::op::PagedAttentionExtension>(ov::OutputVector{Reshape_51953,
                                                                               Reshape_51957,
                                                                               Reshape_51959,
                                                                               key_cache_0,
                                                                               value_cache_0,
                                                                               past_lens,
                                                                               subsequence_begins,
                                                                               block_indices,
                                                                               block_indices_begins,
                                                                               c1,
                                                                               c2,
                                                                               c3,
                                                                               max_context_len,
                                                                               score_aggregation_window,
                                                                               rotated_block_indices,
                                                                               rotation_deltas,
                                                                               rotation_trig_lut,
                                                                               xattention_threshold,
                                                                               xattention_block_size,
                                                                               xattention_stride,
                                                                               sinks});
        auto ShapeOf_51965 = makeOP<opset3::ShapeOf>({Transpose_51955}, {{"output_type", "i64"}});
        auto Gather_51966 = makeOP<opset8::Gather>({ShapeOf_51965, -1, 0}, {{"batch_dims", 0}});
        auto Unsqueeze_51971 = makeOP<opset1::Unsqueeze>({Gather_51966, 0});
        auto Concat_51972 = makeOP<opset1::Concat>({{0l}, {1l}, {-1l}, Unsqueeze_51971}, {{"axis", 0}});
        auto Reshape_51973 =
            makeOP<opset1::Reshape>({PagedAttentionExtension_51962->output(0), Concat_51972}, {{"special_zero", true}});
        auto pref_1_scaled_dot_product_attention_ScaledDotProductAttention =
            makeOP<opset1::Transpose>({Reshape_51973, {0, 2, 1, 3}});

        auto res = std::make_shared<v0::Result>(pref_1_scaled_dot_product_attention_ScaledDotProductAttention);
        model_ref = std::make_shared<ov::Model>(ResultVector{res}, params);

        comparator.disable(FunctionsComparator::PRECISIONS);
        disable_result_friendly_names_check();
        disable_rt_info_check();
    }
}

// todo: split the code to functional blocks as for Qwen-7b model
TEST_F(SDPAToPATest, SDPAToPA_Phi3_mini_4k_instruct) {
    {
        auto beam_idx = make_param(PartialShape{DYN}, element::i32, "beam_idx");
        auto position_ids = make_param(PartialShape{DYN, DYN}, element::i64, "position_ids");
        auto attention_mask = make_param(PartialShape{DYN, DYN}, element::i64, "attention_mask");
        auto input_ids = make_param(PartialShape{DYN, DYN}, element::i64, "input_ids");
        auto params = nodes_to_params({beam_idx, position_ids, attention_mask, input_ids});

        auto ShapeOf = makeOP<opset3::ShapeOf>({input_ids}, {{"output_type", "i64"}});
        auto Gather = makeOP<opset8::Gather>({ShapeOf, {0}, 0}, {{"batch_dims", 0}});
        auto Concat = makeOP<opset1::Concat>({Gather, {32ll}, {0ll}, {96ll}}, {{"axis", 0}});
        auto Broadcast = makeOP<opset3::Broadcast>({0.000000f, Concat}, {{"mode", "numpy"}});

        auto Constant7 = makeConst(element::f32, ov::Shape({1, 1, 3072}), MOCK_VALUE);
        auto Constant8 = makeConst(element::u8, ov::Shape({WEIGHTS, 3072}), MOCK_VALUE);
        auto Convert = makeOP<opset1::Convert>({Constant8}, {{"destination_type", "f16"}});
        auto Constant9 = makeConst(element::u8, ov::Shape({WEIGHTS, 1}), MOCK_VALUE);
        auto Convert1 = makeOP<opset1::Convert>({Constant9}, {{"destination_type", "f16"}});
        auto Subtract = makeOP<opset1::Subtract>({Convert, Convert1}, {{"auto_broadcast", "numpy"}});
        auto Constant10 = makeConst(element::f16, ov::Shape({WEIGHTS, 1}), MOCK_VALUE);
        auto Multiply = makeOP<opset1::Multiply>({Subtract, Constant10}, {{"auto_broadcast", "numpy"}});
        auto Convert2 = makeOP<opset1::Convert>({Multiply}, {{"destination_type", "f32"}});
        auto Convert3 = makeOP<opset1::Convert>({input_ids}, {{"destination_type", "i32"}});
        auto Gather2 = makeOP<opset8::Gather>({Convert2, Convert3, 0}, {{"batch_dims", 0}});
        auto Constant12 = makeConst(element::f32, ov::Shape({1, 1, 3072}), MOCK_VALUE);
        auto Constant13 = makeConst(element::f32, ov::Shape({1, 1, 1}), {1.000000f});
        auto Constant14 = makeConst(element::f32, ov::Shape({1, 1, 1}), {2.000000f});
        auto Power = makeOP<opset1::Power>({Gather2, Constant14}, {{"auto_broadcast", "numpy"}});
        auto ReduceMean = makeOP<opset1::ReduceMean>({Power, {-1}}, {{"keep_dims", true}});
        auto Constant16 = makeConst(element::f32, ov::Shape({1, 1, 1}), {0.000010f});
        auto Add = makeOP<opset1::Add>({ReduceMean, Constant16}, {{"auto_broadcast", "numpy"}});
        auto Sqrt = makeOP<opset1::Sqrt>({Add});
        auto Divide = makeOP<opset1::Divide>({Constant13, Sqrt}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Multiply1 = makeOP<opset1::Multiply>({Gather2, Divide}, {{"auto_broadcast", "numpy"}});
        auto Multiply2 = makeOP<opset1::Multiply>({Constant12, Multiply1}, {{"auto_broadcast", "numpy"}});
        auto Constant17 = makeConst(element::u8, ov::Shape({9216, 3072}), MOCK_VALUE);
        auto Convert4 = makeOP<opset1::Convert>({Constant17}, {{"destination_type", "f16"}});
        auto Constant18 = makeConst(element::u8, ov::Shape({9216, 1}), MOCK_VALUE);
        auto Convert5 = makeOP<opset1::Convert>({Constant18}, {{"destination_type", "f16"}});
        auto Subtract1 = makeOP<opset1::Subtract>({Convert4, Convert5}, {{"auto_broadcast", "numpy"}});
        auto Constant19 = makeConst(element::f16, ov::Shape({9216, 1}), MOCK_VALUE);
        auto Multiply3 = makeOP<opset1::Multiply>({Subtract1, Constant19}, {{"auto_broadcast", "numpy"}});
        auto Convert6 = makeOP<opset1::Convert>({Multiply3}, {{"destination_type", "f32"}});
        auto MatMul = makeOP<opset1::MatMul>({Multiply2, Convert6}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Slice = makeOP<opset8::Slice>({MatMul, {0}, {3072}, {1}, {2}});
        auto Reshape = makeOP<opset1::Reshape>({Slice, {0, 0, 32, 96}}, {{"special_zero", true}});
        auto Transpose = makeOP<opset1::Transpose>({Reshape, {0, 2, 1, 3}});
        auto Constant26 = makeConst(element::f32, ov::Shape({1, 48, 1}), MOCK_VALUE);
        auto Concat1 = makeOP<opset1::Concat>({Gather, {1ll}, {1ll}}, {{"axis", 0}});
        auto Broadcast1 = makeOP<opset3::Broadcast>({Constant26, Concat1}, {{"mode", "bidirectional"}});
        auto Reshape1 = makeOP<opset1::Reshape>({position_ids, {0, 0}}, {{"special_zero", true}});
        auto Unsqueeze = makeOP<opset1::Unsqueeze>({Reshape1, 1});
        auto Convert7 = makeOP<opset1::Convert>({Unsqueeze}, {{"destination_type", "f32"}});
        auto MatMul1 = makeOP<opset1::MatMul>({Broadcast1, Convert7}, {{"transpose_a", false}, {"transpose_b", false}});
        auto Transpose1 = makeOP<opset1::Transpose>({MatMul1, {0, 2, 1}});
        auto Concat2 = makeOP<opset1::Concat>({Transpose1, Transpose1}, {{"axis", -1}});

        auto Cos = makeOP<opset1::Cos>({Concat2});
        auto Sin = makeOP<opset1::Sin>({Concat2});

        auto Unsqueeze1 = makeOP<opset1::Unsqueeze>({Cos, 1});
        auto Multiply4 = makeOP<opset1::Multiply>({Transpose, Unsqueeze1}, {{"auto_broadcast", "numpy"}});
        auto Slice1 = makeOP<opset8::Slice>({Transpose, {48}, {LLONG_MAX}, {1}, {3}});
        auto Constant36 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {-1.000000f});
        auto Multiply5 = makeOP<opset1::Multiply>({Slice1, Constant36}, {{"auto_broadcast", "numpy"}});
        auto Slice2 = makeOP<opset8::Slice>({Transpose, {0}, {48}, {1}, {3}});
        auto Concat3 = makeOP<opset1::Concat>({Multiply5, Slice2}, {{"axis", -1}});
        auto Unsqueeze2 = makeOP<opset1::Unsqueeze>({Sin, 1});
        auto Multiply6 = makeOP<opset1::Multiply>({Concat3, Unsqueeze2}, {{"auto_broadcast", "numpy"}});

        auto Q = makeOP<opset1::Add>({Multiply4, Multiply6}, {{"auto_broadcast", "numpy"}});

        auto ReadValue1 = makeOP<opset6::ReadValue>(
            {Broadcast},
            {{"variable_id", "varid_2"}, {"variable_type", "f32"}, {"variable_shape", PartialShape{DYN, 32, DYN, 96}}});
        auto Gather3 = makeOP<opset8::Gather>({ReadValue1, beam_idx, 0}, {{"batch_dims", 0}});
        auto Slice3 = makeOP<opset8::Slice>({MatMul, {3072}, {6144}, {1}, {2}});
        auto Reshape2 = makeOP<opset1::Reshape>({Slice3, {0, 0, 32, 96}}, {{"special_zero", true}});
        auto Transpose2 = makeOP<opset1::Transpose>({Reshape2, {0, 2, 1, 3}});
        auto Multiply7 = makeOP<opset1::Multiply>({Transpose2, Unsqueeze1}, {{"auto_broadcast", "numpy"}});
        auto Slice4 = makeOP<opset8::Slice>({Transpose2, {48}, {LLONG_MAX}, {1}, {3}});
        auto Constant49 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {-1.000000f});
        auto Multiply8 = makeOP<opset1::Multiply>({Slice4, Constant49}, {{"auto_broadcast", "numpy"}});
        auto Slice5 = makeOP<opset8::Slice>({Transpose2, {0}, {48}, {1}, {3}});
        auto Concat4 = makeOP<opset1::Concat>({Multiply8, Slice5}, {{"axis", -1}});
        auto Multiply9 = makeOP<opset1::Multiply>({Concat4, Unsqueeze2}, {{"auto_broadcast", "numpy"}});
        auto Add2 = makeOP<opset1::Add>({Multiply7, Multiply9}, {{"auto_broadcast", "numpy"}});
        auto K = makeOP<opset1::Concat>({Gather3, Add2}, {{"axis", -2}});
        auto ReadValue2 = makeOP<opset6::ReadValue>(
            {Broadcast},
            {{"variable_id", "varid_3"}, {"variable_type", "f32"}, {"variable_shape", PartialShape{DYN, 32, DYN, 96}}});
        auto Gather4 = makeOP<opset8::Gather>({ReadValue2, beam_idx, 0}, {{"batch_dims", 0}});
        auto Slice6 = makeOP<opset8::Slice>({MatMul, {6144}, {LLONG_MAX}, {1}, {2}});
        auto Reshape3 = makeOP<opset1::Reshape>({Slice6, {0, 0, 32, 96}}, {{"special_zero", true}});
        auto Transpose3 = makeOP<opset1::Transpose>({Reshape3, {0, 2, 1, 3}});
        auto V = makeOP<opset1::Concat>({Gather4, Transpose3}, {{"axis", -2}});
        auto Constant59 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {1.000000f});
        auto Unsqueeze3 = makeOP<opset1::Unsqueeze>({attention_mask, 1});
        auto Unsqueeze4 = makeOP<opset1::Unsqueeze>({Unsqueeze3, 2});
        auto Gather5 = makeOP<opset8::Gather>({ShapeOf, 1, 0}, {{"batch_dims", 0}});
        auto Reshape4 = makeOP<opset1::Reshape>({Gather5, {-1}}, {{"special_zero", false}});
        auto ShapeOf1 = makeOP<opset3::ShapeOf>({attention_mask}, {{"output_type", "i64"}});
        auto Gather6 = makeOP<opset8::Gather>({ShapeOf1, {1}, 0}, {{"batch_dims", 0}});
        auto Concat7 = makeOP<opset1::Concat>({Gather, {1ll}, Reshape4, Gather6}, {{"axis", 0}});
        auto Broadcast2 = makeOP<opset3::Broadcast>({Unsqueeze4, Concat7}, {{"mode", "bidirectional"}});
        auto Convert8 = makeOP<opset1::Convert>({Broadcast2}, {{"destination_type", "f32"}});
        auto Constant67 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {1.000000f});
        auto Multiply10 = makeOP<opset1::Multiply>({Convert8, Constant67}, {{"auto_broadcast", "numpy"}});
        auto Subtract2 = makeOP<opset1::Subtract>({Constant59, Multiply10}, {{"auto_broadcast", "numpy"}});
        auto Convert9 = makeOP<opset1::Convert>({Subtract2}, {{"destination_type", "boolean"}});
        auto Select = makeOP<opset1::Select>({Convert9, -FLT_MAX, Subtract2}, {{"auto_broadcast", "numpy"}});
        auto Convert10 = makeOP<opset1::Convert>({Select}, {{"destination_type", "boolean"}});
        auto Constant69 = makeConst(element::i64, ov::Shape({1, 1}), {1});
        auto ShapeOf2 = makeOP<opset3::ShapeOf>({Gather3}, {{"output_type", "i64"}});
        auto Gather7 = makeOP<opset8::Gather>({ShapeOf2, 2, 0}, {{"batch_dims", 0}});
        auto Add3 = makeOP<opset1::Add>({Gather5, Gather7}, {{"auto_broadcast", "numpy"}});
        auto Subtract3 = makeOP<opset1::Subtract>({Add3, Gather5}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze5 = makeOP<opset1::Unsqueeze>({Subtract3, 0});
        auto Concat8 = makeOP<opset1::Concat>({Reshape4, Unsqueeze5}, {{"axis", 0}});
        auto Broadcast3 = makeOP<opset3::Broadcast>({0.000000f, Concat8}, {{"mode", "numpy"}});
        auto ShapeOf3 = makeOP<opset3::ShapeOf>({Broadcast3}, {{"output_type", "i32"}});
        auto Gather8 = makeOP<opset8::Gather>({ShapeOf3, 1, 0}, {{"batch_dims", 0}});
        auto Convert11 = makeOP<opset1::Convert>({Gather5}, {{"destination_type", "i32"}});
        auto Add4 = makeOP<opset1::Add>({Gather8, Convert11}, {{"auto_broadcast", "numpy"}});
        auto Range = makeOP<opset4::Range>({0, Add4, 1}, {{"output_type", "i32"}});
        auto Unsqueeze6 = makeOP<opset1::Unsqueeze>({Range, 0});
        auto Add5 = makeOP<opset1::Add>({Subtract3, -2046ll}, {{"auto_broadcast", "numpy"}});
        auto Convert12 = makeOP<opset1::Convert>({Add5}, {{"destination_type", "i32"}});
        auto Add6 = makeOP<opset1::Add>({Convert11, Convert12}, {{"auto_broadcast", "numpy"}});
        auto Range1 = makeOP<opset4::Range>({Convert12, Add6, 1}, {{"output_type", "i32"}});
        auto Unsqueeze7 = makeOP<opset1::Unsqueeze>({Range1, 1});
        auto GreaterEqual = makeOP<opset1::GreaterEqual>({Unsqueeze6, Unsqueeze7}, {{"auto_broadcast", "numpy"}});
        auto Range2 = makeOP<opset4::Range>({0, Gather5, 1}, {{"output_type", "f32"}});
        auto Convert13 = makeOP<opset1::Convert>({Range2}, {{"destination_type", "i64"}});
        auto Add7 = makeOP<opset1::Add>({Convert13, {1ll}}, {{"auto_broadcast", "numpy"}});
        auto Reshape5 = makeOP<opset1::Reshape>({Add7, {0, 1}}, {{"special_zero", true}});
        auto Less = makeOP<opset1::Less>({Convert13, Reshape5}, {{"auto_broadcast", "numpy"}});
        auto Broadcast4 = makeOP<opset3::Broadcast>({Reshape4, {2}}, {{"mode", "numpy"}});
        auto Broadcast5 = makeOP<opset3::Broadcast>({-FLT_MAX, Broadcast4}, {{"mode", "numpy"}});
        auto Select1 = makeOP<opset1::Select>({Less, 0.000000f, Broadcast5}, {{"auto_broadcast", "numpy"}});
        auto Concat9 = makeOP<opset1::Concat>({Broadcast3, Select1}, {{"axis", -1}});
        auto ShapeOf4 = makeOP<opset3::ShapeOf>({Concat9}, {{"output_type", "i32"}});
        auto Broadcast6 = makeOP<opset3::Broadcast>({1ll, ShapeOf4}, {{"mode", "numpy"}});
        auto Select2 = makeOP<opset1::Select>({GreaterEqual, Broadcast6, 0ll}, {{"auto_broadcast", "numpy"}});
        auto Subtract4 = makeOP<opset1::Subtract>({Constant69, Select2}, {{"auto_broadcast", "numpy"}});
        auto Convert14 = makeOP<opset1::Convert>({Subtract4}, {{"destination_type", "boolean"}});
        auto Select3 = makeOP<opset1::Select>({Convert14, -FLT_MAX, Concat9}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze8 = makeOP<opset1::Unsqueeze>({Select3, 0});
        auto Unsqueeze9 = makeOP<opset1::Unsqueeze>({Unsqueeze8, 1});
        auto Add8 = makeOP<opset1::Add>({Reshape4, Unsqueeze5}, {{"auto_broadcast", "numpy"}});
        auto Concat10 = makeOP<opset1::Concat>({Gather, {1ll}, Reshape4, Add8}, {{"axis", 0}});
        auto Broadcast7 = makeOP<opset3::Broadcast>({Unsqueeze9, Concat10}, {{"mode", "bidirectional"}});
        auto Select4 = makeOP<opset1::Select>({Convert10, -FLT_MAX, Broadcast7}, {{"auto_broadcast", "numpy"}});
        auto Reshape6 = makeOP<opset1::Reshape>({Gather7, {-1}}, {{"special_zero", false}});
        auto Add9 = makeOP<opset1::Add>({Reshape6, Reshape4}, {{"auto_broadcast", "numpy"}});
        auto attn_mask = makeOP<opset8::Slice>({Select4, {0}, Add9, {1}, {3}});
        auto ScaledDotProductAttention =
            makeOP<v13::ScaledDotProductAttention>({Q, K, V, attn_mask}, {{"causal", false}});
        auto res = make_shared<v0::Result>(ScaledDotProductAttention);
        model = std::make_shared<ov::Model>(OutputVector{res}, params);

        manager.register_pass<ov::pass::SDPAToPagedAttention>();
    }
    {
        auto max_context_len = make_param(PartialShape{}, element::i32, "max_context_len");
        auto block_indices_begins = make_param(PartialShape{DYN}, element::i32, "block_indices_begins");
        auto block_indices = make_param(PartialShape{DYN}, element::i32, "block_indices");
        auto subsequence_begins = make_param(PartialShape{DYN}, element::i32, "subsequence_begins");
        auto past_lens = make_param(PartialShape{DYN}, element::i32, "past_lens");
        auto value_cache_0 = make_param(PartialShape{DYN, 32, 96}, element::f32, "value_cache_0");
        auto key_cache_0 = make_param(PartialShape{DYN, 32, 96}, element::f32, "key_cache_0");
        auto inputs_ids = make_param(PartialShape{DYN}, element::i64, "inputs_ids");
        auto position_ids = make_param(PartialShape{DYN}, element::i64, "position_ids");
        auto score_aggregation_window = makeConst(element::i32, ov::Shape({0}), MOCK_VALUE);
        auto rotated_block_indices = makeConst(element::i32, ov::Shape({0}), {0});
        auto rotation_deltas = makeConst(element::i32, ov::Shape{0}, {0});
        auto rotation_trig_lut = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_threshold = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_block_size = makeConst(element::i32, ov::Shape({}), MOCK_VALUE);
        auto xattention_stride = makeConst(element::i32, ov::Shape({}), MOCK_VALUE);

        auto params = nodes_to_params({max_context_len,
                                       block_indices_begins,
                                       block_indices,
                                       subsequence_begins,
                                       past_lens,
                                       value_cache_0,
                                       key_cache_0,
                                       inputs_ids,
                                       position_ids});

        auto Constant = makeConst(element::f32, ov::Shape({1, 1, 3072}), MOCK_VALUE);
        auto Constant1 = makeConst(element::u8, ov::Shape({WEIGHTS, 3072}), MOCK_VALUE);
        auto Convert = makeOP<opset1::Convert>({Constant1}, {{"destination_type", "f16"}});
        auto Constant2 = makeConst(element::u8, ov::Shape({WEIGHTS, 1}), MOCK_VALUE);
        auto Convert1 = makeOP<opset1::Convert>({Constant2}, {{"destination_type", "f16"}});
        auto Subtract = makeOP<opset1::Subtract>({Convert, Convert1}, {{"auto_broadcast", "numpy"}});
        auto Constant3 = makeConst(element::f16, ov::Shape({WEIGHTS, 1}), MOCK_VALUE);
        auto Multiply = makeOP<opset1::Multiply>({Subtract, Constant3}, {{"auto_broadcast", "numpy"}});
        auto Convert2 = makeOP<opset1::Convert>({Multiply}, {{"destination_type", "f32"}});
        auto Unsqueeze = makeOP<opset1::Unsqueeze>({inputs_ids, 1});
        auto Convert3 = makeOP<opset1::Convert>({Unsqueeze}, {{"destination_type", "i32"}});
        auto Gather = makeOP<opset8::Gather>({Convert2, Convert3, 0}, {{"batch_dims", 0}});
        auto Constant6 = makeConst(element::f32, ov::Shape({1, 1, 1}), {1.000000f});
        auto Constant7 = makeConst(element::f32, ov::Shape({1, 1, 1}), {2.000000f});
        auto Power = makeOP<opset1::Power>({Gather, Constant7}, {{"auto_broadcast", "numpy"}});
        auto ReduceMean = makeOP<opset1::ReduceMean>({Power, {-1}}, {{"keep_dims", true}});
        auto Constant9 = makeConst(element::f32, ov::Shape({1, 1, 1}), {0.000010f});
        auto Add = makeOP<opset1::Add>({ReduceMean, Constant9}, {{"auto_broadcast", "numpy"}});
        auto Sqrt = makeOP<opset1::Sqrt>({Add});
        auto Divide = makeOP<opset1::Divide>({Constant6, Sqrt}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Multiply1 = makeOP<opset1::Multiply>({Gather, Divide}, {{"auto_broadcast", "numpy"}});
        auto Multiply2 = makeOP<opset1::Multiply>({Constant, Multiply1}, {{"auto_broadcast", "numpy"}});
        auto Constant10 = makeConst(element::u8, ov::Shape({9216, 3072}), MOCK_VALUE);
        auto Convert4 = makeOP<opset1::Convert>({Constant10}, {{"destination_type", "f16"}});
        auto Constant11 = makeConst(element::u8, ov::Shape({9216, 1}), MOCK_VALUE);
        auto Convert5 = makeOP<opset1::Convert>({Constant11}, {{"destination_type", "f16"}});
        auto Subtract1 = makeOP<opset1::Subtract>({Convert4, Convert5}, {{"auto_broadcast", "numpy"}});
        auto Constant12 = makeConst(element::f16, ov::Shape({9216, 1}), MOCK_VALUE);
        auto Multiply3 = makeOP<opset1::Multiply>({Subtract1, Constant12}, {{"auto_broadcast", "numpy"}});
        auto Convert6 = makeOP<opset1::Convert>({Multiply3}, {{"destination_type", "f32"}});
        auto MatMul = makeOP<opset1::MatMul>({Multiply2, Convert6}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Slice = makeOP<opset8::Slice>({MatMul, {0}, {3072}, {1}, {2}});
        auto Reshape = makeOP<opset1::Reshape>({Slice, {0, 0, 32, 96}}, {{"special_zero", true}});
        auto Transpose = makeOP<opset1::Transpose>({Reshape, {0, 2, 1, 3}});
        auto Constant19 = makeConst(element::f32, ov::Shape({1, 48, 1}), MOCK_VALUE);
        auto ShapeOf = makeOP<opset3::ShapeOf>({Unsqueeze}, {{"output_type", "i64"}});
        auto Gather1 = makeOP<opset8::Gather>({ShapeOf, {0}, 0}, {{"batch_dims", 0}});
        auto Concat = makeOP<opset1::Concat>({Gather1, {1ll}, {1ll}}, {{"axis", 0}});
        auto Broadcast = makeOP<opset3::Broadcast>({Constant19, Concat}, {{"mode", "bidirectional"}});
        auto Unsqueeze1 = makeOP<opset1::Unsqueeze>({position_ids, 1});
        auto Reshape1 = makeOP<opset1::Reshape>({Unsqueeze1, {0, 0}}, {{"special_zero", true}});
        auto Unsqueeze2 = makeOP<opset1::Unsqueeze>({Reshape1, 1});
        auto Convert7 = makeOP<opset1::Convert>({Unsqueeze2}, {{"destination_type", "f32"}});
        auto MatMul1 = makeOP<opset1::MatMul>({Broadcast, Convert7}, {{"transpose_a", false}, {"transpose_b", false}});
        auto Transpose1 = makeOP<opset1::Transpose>({MatMul1, {0, 2, 1}});
        auto Concat1 = makeOP<opset1::Concat>({Transpose1, Transpose1}, {{"axis", -1}});
        auto Cos = makeOP<opset1::Cos>({Concat1});
        auto Unsqueeze3 = makeOP<opset1::Unsqueeze>({Cos, 1});
        auto Multiply4 = makeOP<opset1::Multiply>({Transpose, Unsqueeze3}, {{"auto_broadcast", "numpy"}});
        auto Slice1 = makeOP<opset8::Slice>({Transpose, {48}, {LLONG_MAX}, {1}, {3}});
        auto Constant33 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {-1.000000f});
        auto Multiply5 = makeOP<opset1::Multiply>({Slice1, Constant33}, {{"auto_broadcast", "numpy"}});
        auto Slice2 = makeOP<opset8::Slice>({Transpose, {0}, {48}, {1}, {3}});
        auto Concat2 = makeOP<opset1::Concat>({Multiply5, Slice2}, {{"axis", -1}});
        auto Sin = makeOP<opset1::Sin>({Concat1});
        auto Unsqueeze4 = makeOP<opset1::Unsqueeze>({Sin, 1});
        auto Multiply6 = makeOP<opset1::Multiply>({Concat2, Unsqueeze4}, {{"auto_broadcast", "numpy"}});
        auto Add1 = makeOP<opset1::Add>({Multiply4, Multiply6}, {{"auto_broadcast", "numpy"}});
        auto Transpose2 = makeOP<opset1::Transpose>({Add1, {0, 2, 1, 3}});
        auto Q = makeOP<opset1::Reshape>({Transpose2, {0, -1}}, {{"special_zero", true}});

        auto Slice3 = makeOP<opset8::Slice>({MatMul, {3072}, {6144}, {1}, {2}});
        auto Reshape3 = makeOP<opset1::Reshape>({Slice3, {0, 0, 32, 96}}, {{"special_zero", true}});
        auto Transpose3 = makeOP<opset1::Transpose>({Reshape3, {0, 2, 1, 3}});
        auto Multiply7 = makeOP<opset1::Multiply>({Transpose3, Unsqueeze3}, {{"auto_broadcast", "numpy"}});
        auto Slice4 = makeOP<opset8::Slice>({Transpose3, {48}, {LLONG_MAX}, {1}, {3}});
        auto Constant51 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {-1.000000f});
        auto Multiply8 = makeOP<opset1::Multiply>({Slice4, Constant51}, {{"auto_broadcast", "numpy"}});
        auto Slice5 = makeOP<opset8::Slice>({Transpose3, {0}, {48}, {1}, {3}});
        auto Concat3 = makeOP<opset1::Concat>({Multiply8, Slice5}, {{"axis", -1}});
        auto Multiply9 = makeOP<opset1::Multiply>({Concat3, Unsqueeze4}, {{"auto_broadcast", "numpy"}});
        auto Add2 = makeOP<opset1::Add>({Multiply7, Multiply9}, {{"auto_broadcast", "numpy"}});
        auto Transpose4 = makeOP<opset1::Transpose>({Add2, {0, 2, 1, 3}});
        auto K = makeOP<opset1::Reshape>({Transpose4, {0, -1}}, {{"special_zero", true}});

        auto Slice6 = makeOP<opset8::Slice>({MatMul, {6144}, {LLONG_MAX}, {1}, {2}});
        auto Reshape5 = makeOP<opset1::Reshape>({Slice6, {0, 0, 32, 96}}, {{"special_zero", true}});
        auto Transpose5 = makeOP<opset1::Transpose>({Reshape5, {0, 2, 1, 3}});
        auto Transpose6 = makeOP<opset1::Transpose>({Transpose5, {0, 2, 1, 3}});
        auto V = makeOP<opset1::Reshape>({Transpose6, {0, -1}}, {{"special_zero", true}});

        auto offset = makeOP<opset1::Convert>({-2046}, {{"destination_type", "i32"}});
        auto sliding_window = makeOP<opset1::Subtract>({2, offset}, {{"auto_broadcast", "numpy"}});

        auto scale = v0::Constant::create(element::f32, {}, {0.102062f});
        auto alibi_slopes = v0::Constant::create(element::f32, Shape{0}, {});
        auto sinks = v0::Constant::create(element::f32, Shape{0, 0, 0, 0}, {});
        auto PagedAttentionExtension =
            std::make_shared<ov::op::PagedAttentionExtension>(OutputVector{Q,
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
                                                                           max_context_len,
                                                                           score_aggregation_window,
                                                                           rotated_block_indices,
                                                                           rotation_deltas,
                                                                           rotation_trig_lut,
                                                                           xattention_threshold,
                                                                           xattention_block_size,
                                                                           xattention_stride,
                                                                           sinks});
        auto ShapeOf1 = makeOP<opset3::ShapeOf>({Transpose6}, {{"output_type", "i64"}});
        auto Gather2 = makeOP<opset8::Gather>({ShapeOf1, -1, 0}, {{"batch_dims", 0}});
        auto Unsqueeze5 = makeOP<opset1::Unsqueeze>({Gather2, 0});
        auto Concat4 = makeOP<opset1::Concat>({{0ll}, {1ll}, {-1ll}, Unsqueeze5}, {{"axis", 0}});
        auto Reshape7 =
            makeOP<opset1::Reshape>({PagedAttentionExtension->output(0), Concat4}, {{"special_zero", true}});
        auto Transpose7 = makeOP<opset1::Transpose>({Reshape7, {0, 2, 1, 3}});
        auto result = makeOP<opset1::Result>({Transpose7});

        model_ref = std::make_shared<ov::Model>(result, params);
        comparator.disable(FunctionsComparator::PRECISIONS);
        disable_result_friendly_names_check();
        disable_rt_info_check();
    }
}

TEST_F(SDPAToPATest, SDPAToPA_Codegen2) {
    {
        auto beam_idx = make_param(PartialShape{DYN}, element::i32, "beam_idx");
        auto position_ids = make_param(PartialShape{DYN, DYN}, element::i64, "position_ids");
        auto attention_mask = make_param(PartialShape{DYN, DYN}, element::i64, "attention_mask");
        auto input_ids = make_param(PartialShape{DYN, DYN}, element::i64, "input_ids");
        auto params = nodes_to_params({beam_idx, position_ids, attention_mask, input_ids});

        auto Constant0 = makeConst(element::f16, ov::Shape({}), {0});
        auto Convert0 = makeOP<opset1::Convert>({Constant0}, {{"destination_type", "f32"}});
        auto ShapeOf0 = makeOP<opset3::ShapeOf>({input_ids}, {{"output_type", "i64"}});
        auto Gather0 = makeOP<opset8::Gather>({ShapeOf0, {0}, 0}, {{"batch_dims", 0}});
        auto Concat0 = makeOP<opset1::Concat>({Gather0, {16l}, {0l}, {256l}}, {{"axis", 0}});
        auto Broadcast0 = makeOP<opset3::Broadcast>({Convert0, Concat0}, {{"mode", "numpy"}});
        auto ReadValue0 = makeOP<opset6::ReadValue>(
            {Broadcast0},
            {{"variable_id", "var1"}, {"variable_type", "f32"}, {"variable_shape", PartialShape{DYN, 16, DYN, 256}}});
        auto Gather1 = makeOP<opset8::Gather>({ReadValue0, beam_idx, 0}, {{"batch_dims", 0}});
        auto Constant1 = makeConst(element::f16, ov::Shape({51200, 4096}), MOCK_VALUE);
        auto Convert1 = makeOP<opset1::Convert>({Constant1}, {{"destination_type", "f32"}});
        auto Reshape0 = makeOP<opset1::Reshape>({input_ids, {-1, 0}}, {{"special_zero", true}});
        auto Convert2 = makeOP<opset1::Convert>({Reshape0}, {{"destination_type", "i32"}});
        auto Gather2 = makeOP<opset8::Gather>({Convert1, Convert2, 0}, {{"batch_dims", 0}});
        auto MVN0 = makeOP<opset6::MVN>({Gather2, {-1}},
                                        {{"eps", 0.000010}, {"normalize_variance", true}, {"eps_mode", "INSIDE_SQRT"}});
        auto Constant2 = makeConst(element::f16, ov::Shape({1, 1, 4096}), MOCK_VALUE);
        auto Convert3 = makeOP<opset1::Convert>({Constant2}, {{"destination_type", "f32"}});
        auto Multiply0 = makeOP<opset1::Multiply>({MVN0, Convert3}, {{"auto_broadcast", "numpy"}});
        auto Constant3 = makeConst(element::f16, ov::Shape({1, 1, 4096}), MOCK_VALUE);
        auto Convert4 = makeOP<opset1::Convert>({Constant3}, {{"destination_type", "f32"}});
        auto Add0 = makeOP<opset1::Add>({Multiply0, Convert4}, {{"auto_broadcast", "numpy"}});
        auto Constant4 = makeConst(element::f16, ov::Shape({12288, 4096}), MOCK_VALUE);
        auto Convert5 = makeOP<opset1::Convert>({Constant4}, {{"destination_type", "f32"}});
        auto MatMul0 = makeOP<opset1::MatMul>({Add0, Convert5}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Reshape1 = makeOP<opset1::Reshape>({MatMul0, {0, 0, 4, -1}}, {{"special_zero", true}});
        auto VariadicSplit0 = makeOP<opset1::VariadicSplit>({Reshape1, -1, {1024, 1024, -1}});
        auto Reshape2 =
            makeOP<opset1::Reshape>({VariadicSplit0->output(0), {0, 0, 0, 4, 256}}, {{"special_zero", true}});
        auto Reshape3 = makeOP<opset1::Reshape>({Reshape2, {0, 0, 16, 256}}, {{"special_zero", true}});
        auto Slice0 = makeOP<opset8::Slice>({Reshape3, {0}, {64}, {1}, {3}});
        auto ShapeOf1 = makeOP<opset3::ShapeOf>({VariadicSplit0->output(0)}, {{"output_type", "i64"}});
        auto Gather3 = makeOP<opset8::Gather>({ShapeOf1, 1, 0}, {{"batch_dims", 0}});
        auto Constant5 = makeConst(element::f16, ov::Shape({}), {0});
        auto Convert6 = makeOP<opset1::Convert>({Constant5}, {{"destination_type", "f32"}});
        auto Concat1 = makeOP<opset1::Concat>({Gather0, {16l}, {0l}, {256l}}, {{"axis", 0}});
        auto Broadcast1 = makeOP<opset3::Broadcast>({Convert6, Concat1}, {{"mode", "numpy"}});
        auto ReadValue1 = makeOP<opset6::ReadValue>(
            {Broadcast1},
            {{"variable_id", "var1"}, {"variable_type", "f32"}, {"variable_shape", PartialShape{DYN, 16, DYN, 256}}});
        auto Gather4 = makeOP<opset8::Gather>({ReadValue1, beam_idx, 0}, {{"batch_dims", 0}});
        auto ShapeOf2 = makeOP<opset3::ShapeOf>({Gather4}, {{"output_type", "i64"}});
        auto Gather5 = makeOP<opset8::Gather>({ShapeOf2, 2, 0}, {{"batch_dims", 0}});
        auto Add1 = makeOP<opset1::Add>({Gather3, Gather5}, {{"auto_broadcast", "numpy"}});
        auto Range0 = makeOP<opset4::Range>({0, Add1, 1}, {{"output_type", "f32"}});
        auto Constant6 = makeConst(element::f16, ov::Shape({1}), {10000});
        auto Convert7 = makeOP<opset1::Convert>({Constant6}, {{"destination_type", "f32"}});
        auto Reshape4 =
            makeOP<opset1::Reshape>({VariadicSplit0->output(2), {0, 0, 0, 4, 256}}, {{"special_zero", true}});
        auto Reshape5 = makeOP<opset1::Reshape>({Reshape4, {0, 0, 16, 256}}, {{"special_zero", true}});
        auto Slice1 = makeOP<opset8::Slice>({Reshape5, {0}, {64}, {1}, {3}});
        auto ShapeOf3 = makeOP<opset3::ShapeOf>({Slice1}, {{"output_type", "i64"}});
        auto Gather6 = makeOP<opset8::Gather>({ShapeOf3, 3, 0}, {{"batch_dims", 0}});
        auto Range1 = makeOP<opset4::Range>({0, Gather6, 2}, {{"output_type", "f32"}});
        auto Convert8 = makeOP<opset1::Convert>({Gather6}, {{"destination_type", "f32"}});
        auto Divide0 = makeOP<opset1::Divide>({Range1, Convert8}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Power0 = makeOP<opset1::Power>({Convert7, Divide0}, {{"auto_broadcast", "numpy"}});
        auto Constant7 = makeConst(element::f16, ov::Shape({1}), {-1});
        auto Convert9 = makeOP<opset1::Convert>({Constant7}, {{"destination_type", "f32"}});
        auto Power1 = makeOP<opset1::Power>({Power0, Convert9}, {{"auto_broadcast", "numpy"}});
        auto Einsum0 = makeOP<opset7::Einsum>({Range0, Power1}, {{"equation", "i,j->ij"}});
        auto Cos0 = makeOP<opset1::Cos>({Einsum0});
        auto Reshape6 = makeOP<opset1::Reshape>({Cos0, {-1, 1}}, {{"special_zero", false}});
        auto Tile0 = makeOP<opset1::Tile>({Reshape6, {1, 2}});
        auto ShapeOf4 = makeOP<opset3::ShapeOf>({Cos0}, {{"output_type", "i64"}});
        auto Gather7 = makeOP<opset8::Gather>({ShapeOf4, {0}, 0}, {{"batch_dims", 0}});
        auto Concat2 = makeOP<opset1::Concat>({Gather7, {-1l}}, {{"axis", 0}});
        auto Reshape7 = makeOP<opset1::Reshape>({Tile0, Concat2}, {{"special_zero", false}});
        auto Unsqueeze0 = makeOP<opset1::Unsqueeze>({Reshape7, 0});
        auto Reshape8 = makeOP<opset1::Reshape>({Gather5, {1}}, {{"special_zero", false}});
        auto Reshape9 = makeOP<opset1::Reshape>({Gather3, {1}}, {{"special_zero", false}});
        auto Add2 = makeOP<opset1::Add>({Reshape9, Reshape8}, {{"auto_broadcast", "numpy"}});
        auto Slice2 = makeOP<opset8::Slice>({Unsqueeze0, Reshape8, Add2, {1}, {1}});
        auto Unsqueeze1 = makeOP<opset1::Unsqueeze>({Slice2, 2});
        auto Multiply1 = makeOP<opset1::Multiply>({Slice0, Unsqueeze1}, {{"auto_broadcast", "numpy"}});
        auto Slice3 = makeOP<opset8::Slice>({Slice0, {1}, {LLONG_MAX}, {2}, {3}});
        auto Convert10 = makeOP<opset1::Convert>({-1}, {{"destination_type", "f32"}});
        auto Multiply2 = makeOP<opset1::Multiply>({Slice3, Convert10}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze2 = makeOP<opset1::Unsqueeze>({Multiply2, -1});
        auto Slice4 = makeOP<opset8::Slice>({Slice0, {0}, {LLONG_MAX}, {2}, {3}});
        auto Unsqueeze3 = makeOP<opset1::Unsqueeze>({Slice4, -1});
        auto Concat3 = makeOP<opset1::Concat>({Unsqueeze2, Unsqueeze3}, {{"axis", -1}});
        auto Reshape10 = makeOP<opset1::Reshape>({Concat3, {0, 0, 16, 64}}, {{"special_zero", true}});
        auto Sin0 = makeOP<opset1::Sin>({Einsum0});
        auto Reshape11 = makeOP<opset1::Reshape>({Sin0, {-1, 1}}, {{"special_zero", false}});
        auto Tile1 = makeOP<opset1::Tile>({Reshape11, {1, 2}});
        auto Reshape12 = makeOP<opset1::Reshape>({Tile1, Concat2}, {{"special_zero", false}});
        auto Unsqueeze4 = makeOP<opset1::Unsqueeze>({Reshape12, 0});
        auto Slice5 = makeOP<opset8::Slice>({Unsqueeze4, Reshape8, Add2, {1}, {1}});
        auto Unsqueeze5 = makeOP<opset1::Unsqueeze>({Slice5, 2});
        auto Multiply3 = makeOP<opset1::Multiply>({Reshape10, Unsqueeze5}, {{"auto_broadcast", "numpy"}});
        auto Add3 = makeOP<opset1::Add>({Multiply1, Multiply3}, {{"auto_broadcast", "numpy"}});
        auto Slice6 = makeOP<opset8::Slice>({Reshape3, {64}, {LLONG_MAX}, {1}, {3}});
        auto Concat4 = makeOP<opset1::Concat>({Add3, Slice6}, {{"axis", -1}});
        auto Transpose0 = makeOP<opset1::Transpose>({Concat4, {0, 2, 1, 3}});
        auto Multiply4 = makeOP<opset1::Multiply>({Slice1, Unsqueeze1}, {{"auto_broadcast", "numpy"}});
        auto Slice7 = makeOP<opset8::Slice>({Slice1, {1}, {LLONG_MAX}, {2}, {3}});
        auto Convert11 = makeOP<opset1::Convert>({-1}, {{"destination_type", "f32"}});
        auto Multiply5 = makeOP<opset1::Multiply>({Slice7, Convert11}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze6 = makeOP<opset1::Unsqueeze>({Multiply5, -1});
        auto Slice8 = makeOP<opset8::Slice>({Slice1, {0}, {LLONG_MAX}, {2}, {3}});
        auto Unsqueeze7 = makeOP<opset1::Unsqueeze>({Slice8, -1});
        auto Concat5 = makeOP<opset1::Concat>({Unsqueeze6, Unsqueeze7}, {{"axis", -1}});
        auto Reshape13 = makeOP<opset1::Reshape>({Concat5, {0, 0, 16, 64}}, {{"special_zero", true}});
        auto Multiply6 = makeOP<opset1::Multiply>({Reshape13, Unsqueeze5}, {{"auto_broadcast", "numpy"}});
        auto Add4 = makeOP<opset1::Add>({Multiply4, Multiply6}, {{"auto_broadcast", "numpy"}});
        auto Slice9 = makeOP<opset8::Slice>({Reshape5, {64}, {LLONG_MAX}, {1}, {3}});
        auto Concat6 = makeOP<opset1::Concat>({Add4, Slice9}, {{"axis", -1}});
        auto Transpose1 = makeOP<opset1::Transpose>({Concat6, {0, 2, 1, 3}});
        auto Concat7 = makeOP<opset1::Concat>({Gather4, Transpose1}, {{"axis", -2}});
        auto Constant8_compressed = makeConst(element::f16, ov::Shape({}), {0});
        auto Convert12 = makeOP<opset1::Convert>({Constant8_compressed}, {{"destination_type", "f32"}});
        auto Concat8 = makeOP<opset1::Concat>({Gather0, {16l}, {0l}, {256l}}, {{"axis", 0}});
        auto Broadcast2 = makeOP<opset3::Broadcast>({Convert12, Concat8}, {{"mode", "numpy"}});
        auto ReadValue2 = makeOP<opset6::ReadValue>(
            {Broadcast2},
            {{"variable_id", "var2"}, {"variable_type", "f32"}, {"variable_shape", PartialShape{DYN, 16, DYN, 256}}});
        auto Gather8 = makeOP<opset8::Gather>({ReadValue2, beam_idx, 0}, {{"batch_dims", 0}});
        auto Reshape14 =
            makeOP<opset1::Reshape>({VariadicSplit0->output(1), {0, 0, 0, 4, 256}}, {{"special_zero", true}});
        auto Reshape15 = makeOP<opset1::Reshape>({Reshape14, {0, 0, 16, 256}}, {{"special_zero", true}});
        auto Transpose2 = makeOP<opset1::Transpose>({Reshape15, {0, 2, 1, 3}});
        auto Concat9 = makeOP<opset1::Concat>({Gather8, Transpose2}, {{"axis", -2}});
        auto Constant9 = makeConst(element::u8, ov::Shape({1, 1, 2048, 2048}), MOCK_VALUE);
        auto Add5 = makeOP<opset1::Add>({Reshape8, Reshape9}, {{"auto_broadcast", "numpy"}});
        auto Subtract0 = makeOP<opset1::Subtract>({Add5, Reshape9}, {{"auto_broadcast", "numpy"}});
        auto Concat10 = makeOP<opset1::Concat>({Subtract0, {0l}}, {{"axis", 0}});
        auto Broadcast3 = makeOP<opset3::Broadcast>({Add5, {2}}, {{"mode", "numpy"}});
        auto Slice10 = makeOP<opset8::Slice>({Constant9, Concat10, Broadcast3, {1, 1}, {2, 3}});
        auto Convert13 = makeOP<opset1::Convert>({Slice10}, {{"destination_type", "boolean"}});
        auto Constant10 = makeConst(element::f16, ov::Shape({}), {0});
        auto Convert14 = makeOP<opset1::Convert>({Constant10}, {{"destination_type", "f32"}});
        auto Select0 = makeOP<opset1::Select>({Convert13, Convert14, -FLT_MAX}, {{"auto_broadcast", "numpy"}});
        auto Constant11 = makeConst(element::f16, ov::Shape({1, 1, 1, 1}), {1});
        auto Convert15 = makeOP<opset1::Convert>({Constant11}, {{"destination_type", "f32"}});
        auto Reshape16 = makeOP<opset1::Reshape>({attention_mask, {0, 0}}, {{"special_zero", true}});
        auto Unsqueeze8 = makeOP<opset1::Unsqueeze>({Reshape16, 1});
        auto Unsqueeze9 = makeOP<opset1::Unsqueeze>({Unsqueeze8, 2});
        auto Convert16 = makeOP<opset1::Convert>({Unsqueeze9}, {{"destination_type", "f32"}});
        auto Constant12 = makeConst(element::f16, ov::Shape({1, 1, 1, 1}), {1});
        auto Convert17 = makeOP<opset1::Convert>({Constant12}, {{"destination_type", "f32"}});
        auto Multiply7 = makeOP<opset1::Multiply>({Convert16, Convert17}, {{"auto_broadcast", "numpy"}});
        auto Subtract1 = makeOP<opset1::Subtract>({Convert15, Multiply7}, {{"auto_broadcast", "numpy"}});
        auto Constant13 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {-FLT_MAX});
        auto Multiply8 = makeOP<opset1::Multiply>({Subtract1, Constant13}, {{"auto_broadcast", "numpy"}});
        auto Minimum0 = makeOP<opset1::Minimum>({Select0, Multiply8}, {{"auto_broadcast", "numpy"}});
        auto ShapeOf5 = makeOP<opset3::ShapeOf>({Minimum0}, {{"output_type", "i64"}});
        auto Gather9 = makeOP<opset8::Gather>({ShapeOf1, {0}, 0}, {{"batch_dims", 0}});
        auto Concat11 = makeOP<opset1::Concat>({Gather9, {1l}, {1l}, {1l}}, {{"axis", 0}});
        auto Maximum0 = makeOP<opset1::Maximum>({ShapeOf5, Concat11}, {{"auto_broadcast", "numpy"}});
        auto Broadcast4 = makeOP<opset3::Broadcast>({Minimum0, Maximum0}, {{"mode", "numpy"}});
        auto ScaledDotProductAttention =
            makeOP<v13::ScaledDotProductAttention>({Transpose0, Concat7, Concat9, Broadcast4}, {{"causal", false}});

        auto res = make_shared<v0::Result>(ScaledDotProductAttention);

        model = std::make_shared<ov::Model>(OutputVector{res}, params);

        manager.register_pass<ov::pass::SDPAToPagedAttention>();
    }
    {
        auto max_context_len = make_param(PartialShape{}, element::i32, "max_context_len");
        auto block_indices_begins = make_param(PartialShape{DYN}, element::i32, "block_indices_begins");
        auto block_indices = make_param(PartialShape{DYN}, element::i32, "block_indices");
        auto subsequence_begins = make_param(PartialShape{DYN}, element::i32, "subsequence_begins");
        auto past_lens = make_param(PartialShape{DYN}, element::i32, "past_lens");
        auto value_cache_0 = make_param(PartialShape{DYN, 16, 256}, element::f32, "value_cache_0");
        auto key_cache_0 = make_param(PartialShape{DYN, 16, 256}, element::f32, "key_cache_0");
        auto input_ids = make_param(PartialShape{DYN}, element::i64, "inputs_ids");
        auto position_ids = make_param(PartialShape{DYN}, element::i64, "position_ids");
        auto score_aggregation_window = makeConst(element::i32, ov::Shape({0}), MOCK_VALUE);
        auto rotated_block_indices = makeConst(element::i32, ov::Shape({0}), {0});
        auto rotation_deltas = makeConst(element::i32, ov::Shape{0}, {0});
        auto rotation_trig_lut = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_threshold = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_block_size = makeConst(element::i32, ov::Shape({}), MOCK_VALUE);
        auto xattention_stride = makeConst(element::i32, ov::Shape({}), MOCK_VALUE);

        auto params = nodes_to_params({max_context_len,
                                       block_indices_begins,
                                       block_indices,
                                       subsequence_begins,
                                       past_lens,
                                       value_cache_0,
                                       key_cache_0,
                                       input_ids,
                                       position_ids});

        auto Constant1 = makeConst(element::f16, ov::Shape({51200, 4096}), MOCK_VALUE);
        auto Convert0 = makeOP<opset1::Convert>({Constant1}, {{"destination_type", "f32"}});
        auto Unsqueeze0 = makeOP<opset1::Unsqueeze>({input_ids, 1});
        auto Reshape0 = makeOP<opset1::Reshape>({Unsqueeze0, {-1, 0}}, {{"special_zero", true}});
        auto Convert1 = makeOP<opset1::Convert>({Reshape0}, {{"destination_type", "i32"}});
        auto Gather0 = makeOP<opset8::Gather>({Convert0, Convert1, 0}, {{"batch_dims", 0}});
        auto MVN0 = makeOP<opset6::MVN>({Gather0, {-1}},
                                        {{"eps", 0.000010}, {"normalize_variance", true}, {"eps_mode", "INSIDE_SQRT"}});
        auto Constant2 = makeConst(element::f16, ov::Shape({1, 1, 4096}), MOCK_VALUE);
        auto Convert2 = makeOP<opset1::Convert>({Constant2}, {{"destination_type", "f32"}});
        auto Multiply0 = makeOP<opset1::Multiply>({MVN0, Convert2}, {{"auto_broadcast", "numpy"}});
        auto Constant3 = makeConst(element::f16, ov::Shape({1, 1, 4096}), MOCK_VALUE);
        auto Convert3 = makeOP<opset1::Convert>({Constant3}, {{"destination_type", "f32"}});
        auto Add0 = makeOP<opset1::Add>({Multiply0, Convert3}, {{"auto_broadcast", "numpy"}});
        auto Constant4 = makeConst(element::f16, ov::Shape({12288, 4096}), MOCK_VALUE);
        auto Convert4 = makeOP<opset1::Convert>({Constant4}, {{"destination_type", "f32"}});
        auto MatMul0 = makeOP<opset1::MatMul>({Add0, Convert4}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Reshape1 = makeOP<opset1::Reshape>({MatMul0, {0, 0, 4, -1}}, {{"special_zero", true}});
        auto VariadicSplit0 = makeOP<opset1::VariadicSplit>({Reshape1, -1, {1024, 1024, -1}});
        auto Reshape2 =
            makeOP<opset1::Reshape>({VariadicSplit0->output(0), {0, 0, 0, 4, 256}}, {{"special_zero", true}});
        auto Reshape3 = makeOP<opset1::Reshape>({Reshape2, {0, 0, 16, 256}}, {{"special_zero", true}});
        auto Slice0 = makeOP<opset8::Slice>({Reshape3, {0}, {64}, {1}, {3}});
        auto Convert5 = makeOP<opset1::Convert>({max_context_len}, {{"destination_type", "i64"}});
        auto Range0 = makeOP<opset4::Range>({0, Convert5, 1}, {{"output_type", "f32"}});
        auto Constant5 = makeConst(element::f16, ov::Shape({1}), {10000});
        auto Convert6 = makeOP<opset1::Convert>({Constant5}, {{"destination_type", "f32"}});
        auto Reshape4 =
            makeOP<opset1::Reshape>({VariadicSplit0->output(2), {0, 0, 0, 4, 256}}, {{"special_zero", true}});
        auto Reshape5 = makeOP<opset1::Reshape>({Reshape4, {0, 0, 16, 256}}, {{"special_zero", true}});
        auto Slice1 = makeOP<opset8::Slice>({Reshape5, {0}, {64}, {1}, {3}});
        auto ShapeOf0 = makeOP<opset3::ShapeOf>({Slice1}, {{"output_type", "i64"}});
        auto Gather1 = makeOP<opset8::Gather>({ShapeOf0, 3, 0}, {{"batch_dims", 0}});
        auto Range1 = makeOP<opset4::Range>({0, Gather1, 2}, {{"output_type", "f32"}});
        auto Convert7 = makeOP<opset1::Convert>({Gather1}, {{"destination_type", "f32"}});
        auto Divide0 = makeOP<opset1::Divide>({Range1, Convert7}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Power0 = makeOP<opset1::Power>({Convert6, Divide0}, {{"auto_broadcast", "numpy"}});
        auto Constant6 = makeConst(element::f16, ov::Shape({1}), {-1});
        auto Convert8 = makeOP<opset1::Convert>({Constant6}, {{"destination_type", "f32"}});
        auto Power1 = makeOP<opset1::Power>({Power0, Convert8}, {{"auto_broadcast", "numpy"}});
        auto Einsum0 = makeOP<opset7::Einsum>({Range0, Power1}, {{"equation", "i,j->ij"}});
        auto Cos0 = makeOP<opset1::Cos>({Einsum0});
        auto Reshape6 = makeOP<opset1::Reshape>({Cos0, {-1, 1}}, {{"special_zero", false}});
        auto Tile0 = makeOP<opset1::Tile>({Reshape6, {1, 2}});
        auto ShapeOf1 = makeOP<opset3::ShapeOf>({Cos0}, {{"output_type", "i64"}});
        auto Gather2 = makeOP<opset8::Gather>({ShapeOf1, {0}, 0}, {{"batch_dims", 0}});
        auto Concat0 = makeOP<opset1::Concat>({Gather2, {-1l}}, {{"axis", 0}});
        auto Reshape7 = makeOP<opset1::Reshape>({Tile0, Concat0}, {{"special_zero", false}});
        auto Unsqueeze1 = makeOP<opset1::Unsqueeze>({Reshape7, 0});
        auto Gather3 = makeOP<opset8::Gather>({Unsqueeze1, position_ids, 1}, {{"batch_dims", 0}});
        auto Transpose0 = makeOP<opset1::Transpose>({Gather3, {1, 0, 2}});
        auto Unsqueeze2 = makeOP<opset1::Unsqueeze>({Transpose0, 2});
        auto Multiply1 = makeOP<opset1::Multiply>({Slice0, Unsqueeze2}, {{"auto_broadcast", "numpy"}});
        auto Slice2 = makeOP<opset8::Slice>({Slice0, {1}, {LLONG_MAX}, {2}, {3}});
        auto Convert9 = makeOP<opset1::Convert>({-1}, {{"destination_type", "f32"}});
        auto Multiply2 = makeOP<opset1::Multiply>({Slice2, Convert9}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze3 = makeOP<opset1::Unsqueeze>({Multiply2, -1});
        auto Slice3 = makeOP<opset8::Slice>({Slice0, {0}, {LLONG_MAX}, {2}, {3}});
        auto Unsqueeze4 = makeOP<opset1::Unsqueeze>({Slice3, -1});
        auto Concat1 = makeOP<opset1::Concat>({Unsqueeze3, Unsqueeze4}, {{"axis", -1}});
        auto Reshape8 = makeOP<opset1::Reshape>({Concat1, {0, 0, 16, 64}}, {{"special_zero", true}});
        auto Sin0 = makeOP<opset1::Sin>({Einsum0});
        auto Reshape9 = makeOP<opset1::Reshape>({Sin0, {-1, 1}}, {{"special_zero", false}});
        auto Tile1 = makeOP<opset1::Tile>({Reshape9, {1, 2}});
        auto Reshape10 = makeOP<opset1::Reshape>({Tile1, Concat0}, {{"special_zero", false}});
        auto Unsqueeze5 = makeOP<opset1::Unsqueeze>({Reshape10, 0});
        auto Gather4 = makeOP<opset8::Gather>({Unsqueeze5, position_ids, 1}, {{"batch_dims", 0}});
        auto Transpose1 = makeOP<opset1::Transpose>({Gather4, {1, 0, 2}});
        auto Unsqueeze6 = makeOP<opset1::Unsqueeze>({Transpose1, 2});
        auto Multiply3 = makeOP<opset1::Multiply>({Reshape8, Unsqueeze6}, {{"auto_broadcast", "numpy"}});
        auto Add1 = makeOP<opset1::Add>({Multiply1, Multiply3}, {{"auto_broadcast", "numpy"}});
        auto Slice4 = makeOP<opset8::Slice>({Reshape3, {64}, {LLONG_MAX}, {1}, {3}});
        auto Concat2 = makeOP<opset1::Concat>({Add1, Slice4}, {{"axis", -1}});
        auto Transpose2 = makeOP<opset1::Transpose>({Concat2, {0, 2, 1, 3}});
        auto Transpose3 = makeOP<opset1::Transpose>({Transpose2, {0, 2, 1, 3}});
        auto Reshape11 = makeOP<opset1::Reshape>({Transpose3, {0, -1}}, {{"special_zero", true}});
        auto Multiply4 = makeOP<opset1::Multiply>({Slice1, Unsqueeze2}, {{"auto_broadcast", "numpy"}});
        auto Slice5 = makeOP<opset8::Slice>({Slice1, {1}, {LLONG_MAX}, {2}, {3}});
        auto Convert10 = makeOP<opset1::Convert>({-1}, {{"destination_type", "f32"}});
        auto Multiply5 = makeOP<opset1::Multiply>({Slice5, Convert10}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze7 = makeOP<opset1::Unsqueeze>({Multiply5, -1});
        auto Slice6 = makeOP<opset8::Slice>({Slice1, {0}, {LLONG_MAX}, {2}, {3}});
        auto Unsqueeze8 = makeOP<opset1::Unsqueeze>({Slice6, -1});
        auto Concat3 = makeOP<opset1::Concat>({Unsqueeze7, Unsqueeze8}, {{"axis", -1}});
        auto Reshape12 = makeOP<opset1::Reshape>({Concat3, {0, 0, 16, 64}}, {{"special_zero", true}});
        auto Multiply6 = makeOP<opset1::Multiply>({Reshape12, Unsqueeze6}, {{"auto_broadcast", "numpy"}});
        auto Add2 = makeOP<opset1::Add>({Multiply4, Multiply6}, {{"auto_broadcast", "numpy"}});
        auto Slice7 = makeOP<opset8::Slice>({Reshape5, {64}, {LLONG_MAX}, {1}, {3}});
        auto Concat4 = makeOP<opset1::Concat>({Add2, Slice7}, {{"axis", -1}});
        auto Transpose4 = makeOP<opset1::Transpose>({Concat4, {0, 2, 1, 3}});
        auto Transpose5 = makeOP<opset1::Transpose>({Transpose4, {0, 2, 1, 3}});
        auto Reshape13 = makeOP<opset1::Reshape>({Transpose5, {0, -1}}, {{"special_zero", true}});
        auto Reshape14 =
            makeOP<opset1::Reshape>({VariadicSplit0->output(1), {0, 0, 0, 4, 256}}, {{"special_zero", true}});
        auto Reshape15 = makeOP<opset1::Reshape>({Reshape14, {0, 0, 16, 256}}, {{"special_zero", true}});
        auto Transpose6 = makeOP<opset1::Transpose>({Reshape15, {0, 2, 1, 3}});
        auto Transpose7 = makeOP<opset1::Transpose>({Transpose6, {0, 2, 1, 3}});
        auto Reshape16 = makeOP<opset1::Reshape>({Transpose7, {0, -1}}, {{"special_zero", true}});

        auto sliding_window = v0::Constant::create(element::i32, {}, {0});
        auto scale = v0::Constant::create(element::f32, {}, {0.062500f});
        auto alibi_slopes_stub = v0::Constant::create(element::f32, Shape{0}, {});
        auto sinks = v0::Constant::create(element::f32, Shape{0, 0, 0, 0}, {});
        auto PagedAttentionExtension =
            std::make_shared<ov::op::PagedAttentionExtension>(OutputVector{Reshape11,
                                                                           Reshape13,
                                                                           Reshape16,
                                                                           key_cache_0,
                                                                           value_cache_0,
                                                                           past_lens,
                                                                           subsequence_begins,
                                                                           block_indices,
                                                                           block_indices_begins,
                                                                           scale,
                                                                           sliding_window,
                                                                           alibi_slopes_stub,
                                                                           max_context_len,
                                                                           score_aggregation_window,
                                                                           rotated_block_indices,
                                                                           rotation_deltas,
                                                                           rotation_trig_lut,
                                                                           xattention_threshold,
                                                                           xattention_block_size,
                                                                           xattention_stride,
                                                                           sinks});
        auto ShapeOf2 = makeOP<opset3::ShapeOf>({Transpose7}, {{"output_type", "i64"}});
        auto Gather5 = makeOP<opset8::Gather>({ShapeOf2, -1, 0}, {{"batch_dims", 0}});
        auto Unsqueeze9 = makeOP<opset1::Unsqueeze>({Gather5, 0});
        auto Concat5 = makeOP<opset1::Concat>({{0l}, {1l}, {-1l}, Unsqueeze9}, {{"axis", 0}});
        auto Reshape17 =
            makeOP<opset1::Reshape>({PagedAttentionExtension->output(0), Concat5}, {{"special_zero", true}});
        auto Transpose8 = makeOP<opset1::Transpose>({Reshape17, {0, 2, 1, 3}});
        auto res = makeOP<opset1::Result>({Transpose8});

        model_ref = std::make_shared<ov::Model>(res, params);

        comparator.disable(FunctionsComparator::PRECISIONS);
        disable_result_friendly_names_check();
        disable_rt_info_check();
    }
}

TEST_F(SDPAToPATest, SDPAToPA_gpt_oss_General) {
    {
        auto beam_idx = make_param(PartialShape{DYN}, element::i32, "beam_idx");
        auto position_ids = make_param(PartialShape{DYN, DYN}, element::i64, "position_ids");
        auto attention_mask = make_param(PartialShape{DYN, DYN}, element::i64, "attention_mask");
        auto input_ids = make_param(PartialShape{DYN, DYN}, element::i64, "input_ids");
        auto params = nodes_to_params({beam_idx, position_ids, attention_mask, input_ids});

        auto ShapeOf0 = makeOP<v3::ShapeOf>({input_ids}, {{"output_type", "i64"}});
        auto Gather0 = makeOP<v8::Gather>({ShapeOf0, {0}, 0}, {{"batch_dims", 0}});
        auto Constant0 = makeConst(element::u8,
                                   ov::Shape({
                                       201088,
                                       2880,
                                   }),
                                   MOCK_VALUE);
        auto Convert0 = makeOP<v0::Convert>({Constant0}, {{"destination_type", "f16"}});
        auto Constant1 = makeConst(element::u8,
                                   ov::Shape({
                                       201088,
                                       1,
                                   }),
                                   MOCK_VALUE);
        auto Convert1 = makeOP<v0::Convert>({Constant1}, {{"destination_type", "f16"}});
        auto Subtract0 = makeOP<v1::Subtract>({Convert0, Convert1}, {{"auto_broadcast", "numpy"}});
        auto Constant2 = makeConst(element::f16,
                                   ov::Shape({
                                       201088,
                                       1,
                                   }),
                                   MOCK_VALUE);
        auto Multiply0 = makeOP<v1::Multiply>({Subtract0, Constant2}, {{"auto_broadcast", "numpy"}});
        auto Convert2 = makeOP<v0::Convert>({Multiply0}, {{"destination_type", "f32"}});
        auto Convert3 = makeOP<v0::Convert>({input_ids}, {{"destination_type", "i32"}});
        auto Gather1 = makeOP<v8::Gather>({Convert2, Convert3, 0}, {{"batch_dims", 0}});
        auto Constant3 = makeConst(element::f32,
                                   ov::Shape({
                                       1,
                                       1,
                                       2880,
                                   }),
                                   MOCK_VALUE);
        auto Constant4 = makeConst(element::f32,
                                   ov::Shape({
                                       1,
                                       1,
                                       1,
                                   }),
                                   {1.000000f});
        auto Constant5 = makeConst(element::f32,
                                   ov::Shape({
                                       1,
                                       1,
                                       1,
                                   }),
                                   {2.000000f});
        auto Power0 = makeOP<v1::Power>({Gather1, Constant5}, {{"auto_broadcast", "numpy"}});
        auto ReduceMean0 = makeOP<v1::ReduceMean>({Power0, {-1}}, {{"keep_dims", true}});
        auto Constant6 = makeConst(element::f32,
                                   ov::Shape({
                                       1,
                                       1,
                                       1,
                                   }),
                                   {0.000010f});
        auto Add0 = makeOP<v1::Add>({ReduceMean0, Constant6}, {{"auto_broadcast", "numpy"}});
        auto Sqrt0 = makeOP<v0::Sqrt>({Add0});
        auto Divide0 = makeOP<v1::Divide>({Constant4, Sqrt0}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Multiply1 = makeOP<v1::Multiply>({Gather1, Divide0}, {{"auto_broadcast", "numpy"}});
        auto Multiply2 = makeOP<v1::Multiply>({Constant3, Multiply1}, {{"auto_broadcast", "numpy"}});
        auto Constant7 = makeConst(element::u8,
                                   ov::Shape({
                                       4096,
                                       2880,
                                   }),
                                   MOCK_VALUE);
        auto Convert4 = makeOP<v0::Convert>({Constant7}, {{"destination_type", "f16"}});
        auto Constant8 = makeConst(element::u8,
                                   ov::Shape({
                                       4096,
                                       1,
                                   }),
                                   MOCK_VALUE);
        auto Convert5 = makeOP<v0::Convert>({Constant8}, {{"destination_type", "f16"}});
        auto Subtract1 = makeOP<v1::Subtract>({Convert4, Convert5}, {{"auto_broadcast", "numpy"}});
        auto Constant9 = makeConst(element::f16,
                                   ov::Shape({
                                       4096,
                                       1,
                                   }),
                                   MOCK_VALUE);
        auto Multiply3 = makeOP<v1::Multiply>({Subtract1, Constant9}, {{"auto_broadcast", "numpy"}});
        auto Convert6 = makeOP<v0::Convert>({Multiply3}, {{"destination_type", "f32"}});
        auto MatMul0 = makeOP<v0::MatMul>({Multiply2, Convert6}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Constant10 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        1,
                                        4096,
                                    }),
                                    MOCK_VALUE);
        auto Add1 = makeOP<v1::Add>({MatMul0, Constant10}, {{"auto_broadcast", "numpy"}});
        auto Reshape0 = makeOP<v1::Reshape>({Add1, {0, 0, 64, 64}}, {{"special_zero", true}});
        auto Transpose0 = makeOP<v1::Transpose>({Reshape0, {0, 2, 1, 3}});
        auto ShapeOf1 = makeOP<v3::ShapeOf>({Transpose0}, {{"output_type", "i32"}});
        auto Gather2 = makeOP<v8::Gather>({ShapeOf1, -1, {0}}, {{"batch_dims", 0}});
        auto Divide1 = makeOP<v1::Divide>({Gather2, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Mod0 = makeOP<v1::Mod>({Gather2, 2}, {{"auto_broadcast", "numpy"}});
        auto Greater0 = makeOP<v1::Greater>({Mod0, {0}}, {{"auto_broadcast", "numpy"}});
        auto Convert7 = makeOP<v0::Convert>({Greater0}, {{"destination_type", "i32"}});
        auto Add2 = makeOP<v1::Add>({Divide1, Convert7}, {{"auto_broadcast", "numpy"}});
        auto Concat0 = makeOP<v0::Concat>({Add2, {-1}}, {{"axis", 0}});
        auto VariadicSplit0 = makeOP<v1::VariadicSplit>({Transpose0, -1, Concat0});
        auto Constant11 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        32,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto ShapeOf2 = makeOP<v3::ShapeOf>({position_ids}, {{"output_type", "i64"}});
        auto Gather3 = makeOP<v8::Gather>({ShapeOf2, {0}, 0}, {{"batch_dims", 0}});
        auto Concat1 = makeOP<v0::Concat>({Gather3, {1l}, {1l}}, {{"axis", 0}});
        auto Broadcast0 = makeOP<v3::Broadcast>({Constant11, Concat1}, {{"mode", "bidirectional"}});
        auto Unsqueeze0 = makeOP<v0::Unsqueeze>({position_ids, 1});
        auto Convert8 = makeOP<v0::Convert>({Unsqueeze0}, {{"destination_type", "f32"}});
        auto MatMul1 = makeOP<v0::MatMul>({Broadcast0, Convert8}, {{"transpose_a", false}, {"transpose_b", false}});
        auto Transpose1 = makeOP<v1::Transpose>({MatMul1, {0, 2, 1}});
        auto Cos0 = makeOP<v0::Cos>({Transpose1});
        auto Constant12 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        1,
                                        1,
                                    }),
                                    {1.346574f});
        auto Multiply4 = makeOP<v1::Multiply>({Cos0, Constant12}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze1 = makeOP<v0::Unsqueeze>({Multiply4, 1});
        auto Multiply5 = makeOP<v1::Multiply>({VariadicSplit0->output(0), Unsqueeze1}, {{"auto_broadcast", "numpy"}});
        auto Sin0 = makeOP<v0::Sin>({Transpose1});
        auto Constant13 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        1,
                                        1,
                                    }),
                                    {1.346574f});
        auto Multiply6 = makeOP<v1::Multiply>({Sin0, Constant13}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze2 = makeOP<v0::Unsqueeze>({Multiply6, 1});
        auto Multiply7 = makeOP<v1::Multiply>({VariadicSplit0->output(1), Unsqueeze2}, {{"auto_broadcast", "numpy"}});
        auto Subtract2 = makeOP<v1::Subtract>({Multiply5, Multiply7}, {{"auto_broadcast", "numpy"}});
        auto Multiply8 = makeOP<v1::Multiply>({VariadicSplit0->output(1), Unsqueeze1}, {{"auto_broadcast", "numpy"}});
        auto Multiply9 = makeOP<v1::Multiply>({VariadicSplit0->output(0), Unsqueeze2}, {{"auto_broadcast", "numpy"}});
        auto Add3 = makeOP<v1::Add>({Multiply8, Multiply9}, {{"auto_broadcast", "numpy"}});
        auto Concat2 = makeOP<v0::Concat>({Subtract2, Add3}, {{"axis", -1}});
        auto Concat3 = makeOP<v0::Concat>({Gather0, {8l}, {0l}, {64l}}, {{"axis", 0}});
        auto Broadcast1 = makeOP<v3::Broadcast>({0.000000f, Concat3}, {{"mode", "numpy"}});
        auto ReadValue0 = makeOP<v6::ReadValue>(
            {Broadcast1},
            {{"variable_id", "var1"}, {"variable_type", "f32"}, {"variable_shape", PartialShape{DYN, 8, DYN, 64}}});
        auto Gather4 = makeOP<v8::Gather>({ReadValue0, beam_idx, 0}, {{"batch_dims", 0}});
        auto Constant14 = makeConst(element::u8,
                                    ov::Shape({
                                        512,
                                        2880,
                                    }),
                                    MOCK_VALUE);
        auto Convert9 = makeOP<v0::Convert>({Constant14}, {{"destination_type", "f16"}});
        auto Constant15 = makeConst(element::u8,
                                    ov::Shape({
                                        512,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto Convert10 = makeOP<v0::Convert>({Constant15}, {{"destination_type", "f16"}});
        auto Subtract3 = makeOP<v1::Subtract>({Convert9, Convert10}, {{"auto_broadcast", "numpy"}});
        auto Constant16 = makeConst(element::f16,
                                    ov::Shape({
                                        512,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto Multiply10 = makeOP<v1::Multiply>({Subtract3, Constant16}, {{"auto_broadcast", "numpy"}});
        auto Convert11 = makeOP<v0::Convert>({Multiply10}, {{"destination_type", "f32"}});
        auto MatMul2 = makeOP<v0::MatMul>({Multiply2, Convert11}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Reshape1 = makeOP<v1::Reshape>({MatMul2, {0, 0, 8, 64}}, {{"special_zero", true}});
        auto Transpose2 = makeOP<v1::Transpose>({Reshape1, {0, 2, 1, 3}});
        auto ShapeOf3 = makeOP<v3::ShapeOf>({Transpose2}, {{"output_type", "i32"}});
        auto Gather5 = makeOP<v8::Gather>({ShapeOf3, -1, {0}}, {{"batch_dims", 0}});
        auto Divide2 = makeOP<v1::Divide>({Gather5, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Mod1 = makeOP<v1::Mod>({Gather5, 2}, {{"auto_broadcast", "numpy"}});
        auto Greater1 = makeOP<v1::Greater>({Mod1, {0}}, {{"auto_broadcast", "numpy"}});
        auto Convert12 = makeOP<v0::Convert>({Greater1}, {{"destination_type", "i32"}});
        auto Add4 = makeOP<v1::Add>({Divide2, Convert12}, {{"auto_broadcast", "numpy"}});
        auto Concat4 = makeOP<v0::Concat>({Add4, {-1}}, {{"axis", 0}});
        auto VariadicSplit1 = makeOP<v1::VariadicSplit>({Transpose2, -1, Concat4});
        auto Multiply11 = makeOP<v1::Multiply>({VariadicSplit1->output(0), Unsqueeze1}, {{"auto_broadcast", "numpy"}});
        auto Multiply12 = makeOP<v1::Multiply>({VariadicSplit1->output(1), Unsqueeze2}, {{"auto_broadcast", "numpy"}});
        auto Subtract4 = makeOP<v1::Subtract>({Multiply11, Multiply12}, {{"auto_broadcast", "numpy"}});
        auto Multiply13 = makeOP<v1::Multiply>({VariadicSplit1->output(1), Unsqueeze1}, {{"auto_broadcast", "numpy"}});
        auto Multiply14 = makeOP<v1::Multiply>({VariadicSplit1->output(0), Unsqueeze2}, {{"auto_broadcast", "numpy"}});
        auto Add5 = makeOP<v1::Add>({Multiply13, Multiply14}, {{"auto_broadcast", "numpy"}});
        auto Concat5 = makeOP<v0::Concat>({Subtract4, Add5}, {{"axis", -1}});
        auto Concat6 = makeOP<v0::Concat>({Gather4, Concat5}, {{"axis", -2}});
        auto Unsqueeze3 = makeOP<v0::Unsqueeze>({Concat6, 2});
        auto ShapeOf4 = makeOP<v3::ShapeOf>({Concat6}, {{"output_type", "i64"}});
        auto Gather6 = makeOP<v8::Gather>({ShapeOf4, {0, 1}, 0}, {{"batch_dims", 0}});
        auto Gather7 = makeOP<v8::Gather>({ShapeOf4, {2, 3}, 0}, {{"batch_dims", 0}});
        auto Concat7 = makeOP<v0::Concat>({Gather6, {8l}, Gather7}, {{"axis", 0}});
        auto Broadcast2 = makeOP<v3::Broadcast>({Unsqueeze3, Concat7}, {{"mode", "bidirectional"}});
        auto Reshape2 = makeOP<v1::Reshape>({Broadcast2, {0, 64, -1, 64}}, {{"special_zero", true}});
        auto Concat8 = makeOP<v0::Concat>({Gather0, {8l}, {0l}, {64l}}, {{"axis", 0}});
        auto Broadcast3 = makeOP<v3::Broadcast>({0.0f, Concat8}, {{"mode", "numpy"}});
        auto ReadValue1 = makeOP<v6::ReadValue>(
            {Broadcast3},
            {{"variable_id", "var2"}, {"variable_type", "f32"}, {"variable_shape", PartialShape{DYN, 8, DYN, 64}}});
        auto Gather8 = makeOP<v8::Gather>({ReadValue1, beam_idx, 0}, {{"batch_dims", 0}});
        auto Constant17 = makeConst(element::u8,
                                    ov::Shape({
                                        512,
                                        2880,
                                    }),
                                    MOCK_VALUE);
        auto Convert13 = makeOP<v0::Convert>({Constant17}, {{"destination_type", "f16"}});
        auto Constant18 = makeConst(element::u8,
                                    ov::Shape({
                                        512,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto Convert14 = makeOP<v0::Convert>({Constant18}, {{"destination_type", "f16"}});
        auto Subtract5 = makeOP<v1::Subtract>({Convert13, Convert14}, {{"auto_broadcast", "numpy"}});
        auto Constant19 = makeConst(element::f16,
                                    ov::Shape({
                                        512,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto Multiply15 = makeOP<v1::Multiply>({Subtract5, Constant19}, {{"auto_broadcast", "numpy"}});
        auto Convert15 = makeOP<v0::Convert>({Multiply15}, {{"destination_type", "f32"}});
        auto MatMul3 = makeOP<v0::MatMul>({Multiply2, Convert15}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Constant20 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        1,
                                        512,
                                    }),
                                    MOCK_VALUE);
        auto Add6 = makeOP<v1::Add>({MatMul3, Constant20}, {{"auto_broadcast", "numpy"}});
        auto Reshape3 = makeOP<v1::Reshape>({Add6, {0, 0, 8, 64}}, {{"special_zero", true}});
        auto Transpose3 = makeOP<v1::Transpose>({Reshape3, {0, 2, 1, 3}});
        auto Concat9 = makeOP<v0::Concat>({Gather8, Transpose3}, {{"axis", -2}});
        auto Unsqueeze4 = makeOP<v0::Unsqueeze>({Concat9, 2});
        auto Broadcast4 = makeOP<v3::Broadcast>({Unsqueeze4, Concat7}, {{"mode", "bidirectional"}});
        auto Reshape4 = makeOP<v1::Reshape>({Broadcast4, {0, 64, -1, 64}}, {{"special_zero", true}});
        auto Constant21 = makeConst(element::boolean, ov::Shape({}), {1});
        auto Constant22 = makeConst(element::boolean, ov::Shape({}), {1});
        auto Gather9 = makeOP<v8::Gather>({ShapeOf2, 1, 0}, {{"batch_dims", 0}});
        auto Reshape5 = makeOP<v1::Reshape>({Gather9, {1}}, {{"special_zero", false}});
        auto Squeeze0 = makeOP<v0::Squeeze>({Reshape5, 0});
        auto ShapeOf5 = makeOP<v3::ShapeOf>({Gather4}, {{"output_type", "i64"}});
        auto Gather10 = makeOP<v8::Gather>({ShapeOf5, 2, 0}, {{"batch_dims", 0}});
        auto Add7 = makeOP<v1::Add>({Squeeze0, Gather10}, {{"auto_broadcast", "numpy"}});
        auto Range0 = makeOP<v4::Range>({0, Add7, 1}, {{"output_type", "i64"}});
        auto Unsqueeze5 = makeOP<v0::Unsqueeze>({Range0, 0});
        auto Unsqueeze6 = makeOP<v0::Unsqueeze>({Unsqueeze5, 1});
        auto Unsqueeze7 = makeOP<v0::Unsqueeze>({Unsqueeze6, 2});
        auto Convert16 = makeOP<v0::Convert>({Unsqueeze7}, {{"destination_type", "f32"}});
        auto Add8 = makeOP<v1::Add>({Gather10, Gather9}, {{"auto_broadcast", "numpy"}});
        auto Range1 = makeOP<v4::Range>({Gather10, Add8, 1}, {{"output_type", "f32"}});
        auto Unsqueeze8 = makeOP<v0::Unsqueeze>({Range1, 0});
        auto Unsqueeze9 = makeOP<v0::Unsqueeze>({Unsqueeze8, 1});
        auto Unsqueeze10 = makeOP<v0::Unsqueeze>({Unsqueeze9, 3});
        auto Constant23 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        1,
                                        1,
                                        1,
                                    }),
                                    {-128.000000f});
        auto Add9 = makeOP<v1::Add>({Unsqueeze10, Constant23}, {{"auto_broadcast", "numpy"}});
        auto Greater2 = makeOP<v1::Greater>({Convert16, Add9}, {{"auto_broadcast", "numpy"}});
        auto BitwiseAnd0 = makeOP<v13::BitwiseAnd>({Constant22, Greater2}, {{"auto_broadcast", "numpy"}});
        auto LessEqual0 = makeOP<v1::LessEqual>({Convert16, Unsqueeze10}, {{"auto_broadcast", "numpy"}});
        auto BitwiseAnd1 = makeOP<v13::BitwiseAnd>({BitwiseAnd0, LessEqual0}, {{"auto_broadcast", "numpy"}});
        auto BitwiseAnd2 = makeOP<v13::BitwiseAnd>({Constant21, BitwiseAnd1}, {{"auto_broadcast", "numpy"}});
        auto Convert17 = makeOP<v0::Convert>({attention_mask}, {{"destination_type", "boolean"}});
        auto ShapeOf6 = makeOP<v3::ShapeOf>({Convert17}, {{"output_type", "i32"}});
        auto ReduceProd0 = makeOP<v1::ReduceProd>({ShapeOf6, 0}, {{"keep_dims", true}});
        auto Concat10 = makeOP<v0::Concat>({ReduceProd0, {-1}}, {{"axis", 0}});
        auto Reshape6 = makeOP<v1::Reshape>({Convert17, Concat10}, {{"special_zero", true}});
        auto Convert18 = makeOP<v0::Convert>({Unsqueeze7}, {{"destination_type", "i32"}});
        auto Squeeze1 = makeOP<v0::Squeeze>({Gather3});
        auto Range2 = makeOP<v4::Range>({0, Squeeze1, 1}, {{"output_type", "i64"}});
        auto Unsqueeze11 = makeOP<v0::Unsqueeze>({Range2, 1});
        auto Unsqueeze12 = makeOP<v0::Unsqueeze>({Unsqueeze11, 2});
        auto Unsqueeze13 = makeOP<v0::Unsqueeze>({Unsqueeze12, 3});
        auto Convert19 = makeOP<v0::Convert>({Unsqueeze13}, {{"destination_type", "i32"}});
        auto Split0 = makeOP<v1::Split>({ShapeOf6, 0}, {{"num_splits", 2}});
        auto Multiply16 = makeOP<v1::Multiply>({Convert19, Split0->output(1)}, {{"auto_broadcast", "numpy"}});
        auto Add10 = makeOP<v1::Add>({Convert18, Multiply16}, {{"auto_broadcast", "numpy"}});
        auto Gather11 = makeOP<v8::Gather>({Reshape6, Add10, 0}, {{"batch_dims", 0}});
        auto Reshape7 = makeOP<v1::Reshape>({Gather11, {-1}}, {{"special_zero", false}});
        auto ShapeOf7 = makeOP<v3::ShapeOf>({Add10}, {{"output_type", "i32"}});
        auto Reshape8 = makeOP<v1::Reshape>({Reshape7, ShapeOf7}, {{"special_zero", false}});
        auto BitwiseAnd3 = makeOP<v13::BitwiseAnd>({BitwiseAnd2, Reshape8}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze14 = makeOP<v0::Unsqueeze>({Add7, 0});
        auto Concat11 = makeOP<v0::Concat>({Gather3, {1l}, Reshape5, Unsqueeze14}, {{"axis", 0}});
        auto Broadcast5 = makeOP<v3::Broadcast>({BitwiseAnd3, Concat11}, {{"mode", "bidirectional"}});
        auto Select0 = makeOP<v1::Select>({Broadcast5, 0.000000f, -65504.000000f}, {{"auto_broadcast", "numpy"}});
        auto Reshape9 = makeOP<v1::Reshape>({Gather10, {1}}, {{"special_zero", false}});
        auto Add11 = makeOP<v1::Add>({Reshape9, Reshape5}, {{"auto_broadcast", "numpy"}});
        auto Slice0 = makeOP<v8::Slice>({Select0, {0}, Add11, {1}, {3}});
        auto Constant24 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        64,
                                        1,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto ScaledDotProductAttention =
            makeOP<v13::ScaledDotProductAttention>({Concat2, Reshape2, Reshape4, Slice0, 0.125000f, Constant24},
                                                   {{"causal", false}});

        auto res = make_shared<v0::Result>(ScaledDotProductAttention);

        model = std::make_shared<ov::Model>(OutputVector{res}, params);

        manager.register_pass<ov::pass::SDPAToPagedAttention>();
    }

    {
        auto max_context_len = make_param(PartialShape{}, element::i32, "max_context_len");
        auto block_indices_begins = make_param(PartialShape{DYN}, element::i32, "block_indices_begins");
        auto block_indices = make_param(PartialShape{DYN}, element::i32, "block_indices");
        auto subsequence_begins = make_param(PartialShape{DYN}, element::i32, "subsequence_begins");
        auto past_lens = make_param(PartialShape{DYN}, element::i32, "past_lens");
        auto value_cache_0 = make_param(PartialShape{DYN, 8, 64}, element::f32, "value_cache_0");
        auto key_cache_0 = make_param(PartialShape{DYN, 8, 64}, element::f32, "key_cache_0");
        auto input_ids = make_param(PartialShape{DYN}, element::i64, "inputs_ids");
        auto position_ids = make_param(PartialShape{DYN}, element::i64, "position_ids");

        auto score_aggregation_window = makeConst(element::i32, ov::Shape({0}), MOCK_VALUE);
        auto rotated_block_indices = makeConst(element::i32, ov::Shape({0}), {0});
        auto rotation_deltas = makeConst(element::i32, ov::Shape{0}, {0});
        auto rotation_trig_lut = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_threshold = makeConst(element::f32, ov::Shape({0}), {0});
        auto xattention_block_size = makeConst(element::i32, ov::Shape({}), {0});
        auto xattention_stride = makeConst(element::i32, ov::Shape({}), {0});

        auto params = nodes_to_params({max_context_len,
                                       block_indices_begins,
                                       block_indices,
                                       subsequence_begins,
                                       past_lens,
                                       value_cache_0,
                                       key_cache_0,
                                       input_ids,
                                       position_ids});

        auto Constant0 = makeConst(element::f32,
                                   ov::Shape({
                                       1,
                                       1,
                                       2880,
                                   }),
                                   MOCK_VALUE);
        auto Constant1 = makeConst(element::u8,
                                   ov::Shape({
                                       201088,
                                       2880,
                                   }),
                                   MOCK_VALUE);
        auto Convert0 = makeOP<v0::Convert>({Constant1}, {{"destination_type", "f16"}});
        auto Constant2 = makeConst(element::u8,
                                   ov::Shape({
                                       201088,
                                       1,
                                   }),
                                   MOCK_VALUE);
        auto Convert1 = makeOP<v0::Convert>({Constant2}, {{"destination_type", "f16"}});
        auto Subtract0 = makeOP<v1::Subtract>({Convert0, Convert1}, {{"auto_broadcast", "numpy"}});
        auto Constant3 = makeConst(element::f16,
                                   ov::Shape({
                                       201088,
                                       1,
                                   }),
                                   MOCK_VALUE);
        auto Multiply0 = makeOP<v1::Multiply>({Subtract0, Constant3}, {{"auto_broadcast", "numpy"}});
        auto Convert2 = makeOP<v0::Convert>({Multiply0}, {{"destination_type", "f32"}});
        auto Unsqueeze0 = makeOP<v0::Unsqueeze>({input_ids, 1});
        auto Convert3 = makeOP<v0::Convert>({Unsqueeze0}, {{"destination_type", "i32"}});
        auto Gather0 = makeOP<v8::Gather>({Convert2, Convert3, 0}, {{"batch_dims", 0}});
        auto Constant4 = makeConst(element::f32,
                                   ov::Shape({
                                       1,
                                       1,
                                       2880,
                                   }),
                                   MOCK_VALUE);
        auto Constant5 = makeConst(element::f32,
                                   ov::Shape({
                                       1,
                                       1,
                                       1,
                                   }),
                                   {1.000000f});
        auto Constant6 = makeConst(element::f32,
                                   ov::Shape({
                                       1,
                                       1,
                                       1,
                                   }),
                                   {2.000000f});
        auto Power0 = makeOP<v1::Power>({Gather0, Constant6}, {{"auto_broadcast", "numpy"}});
        auto ReduceMean0 = makeOP<v1::ReduceMean>({Power0, {-1}}, {{"keep_dims", true}});
        auto Constant7 = makeConst(element::f32,
                                   ov::Shape({
                                       1,
                                       1,
                                       1,
                                   }),
                                   {0.000010f});
        auto Add0 = makeOP<v1::Add>({ReduceMean0, Constant7}, {{"auto_broadcast", "numpy"}});
        auto Sqrt0 = makeOP<v0::Sqrt>({Add0});
        auto Divide0 = makeOP<v1::Divide>({Constant5, Sqrt0}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Multiply1 = makeOP<v1::Multiply>({Gather0, Divide0}, {{"auto_broadcast", "numpy"}});
        auto Multiply2 = makeOP<v1::Multiply>({Constant4, Multiply1}, {{"auto_broadcast", "numpy"}});
        auto Constant8 = makeConst(element::u8,
                                   ov::Shape({
                                       4096,
                                       2880,
                                   }),
                                   MOCK_VALUE);
        auto Convert4 = makeOP<v0::Convert>({Constant8}, {{"destination_type", "f16"}});
        auto Constant9 = makeConst(element::u8,
                                   ov::Shape({
                                       4096,
                                       1,
                                   }),
                                   MOCK_VALUE);
        auto Convert5 = makeOP<v0::Convert>({Constant9}, {{"destination_type", "f16"}});
        auto Subtract1 = makeOP<v1::Subtract>({Convert4, Convert5}, {{"auto_broadcast", "numpy"}});
        auto Constant10 = makeConst(element::f16,
                                    ov::Shape({
                                        4096,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto Multiply3 = makeOP<v1::Multiply>({Subtract1, Constant10}, {{"auto_broadcast", "numpy"}});
        auto Convert6 = makeOP<v0::Convert>({Multiply3}, {{"destination_type", "f32"}});
        auto MatMul0 = makeOP<v0::MatMul>({Multiply2, Convert6}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Constant11 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        1,
                                        4096,
                                    }),
                                    MOCK_VALUE);
        auto Add1 = makeOP<v1::Add>({MatMul0, Constant11}, {{"auto_broadcast", "numpy"}});
        auto Reshape0 = makeOP<v1::Reshape>({Add1, {0, 0, 64, 64}}, {{"special_zero", true}});
        auto Transpose0 = makeOP<v1::Transpose>({Reshape0, {0, 2, 1, 3}});
        auto ShapeOf0 = makeOP<v3::ShapeOf>({Transpose0}, {{"output_type", "i32"}});
        auto Gather1 = makeOP<v8::Gather>({ShapeOf0, -1, {0}}, {{"batch_dims", 0}});
        auto Divide1 = makeOP<v1::Divide>({Gather1, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Mod0 = makeOP<v1::Mod>({Gather1, 2}, {{"auto_broadcast", "numpy"}});
        auto Greater0 = makeOP<v1::Greater>({Mod0, {0}}, {{"auto_broadcast", "numpy"}});
        auto Convert7 = makeOP<v0::Convert>({Greater0}, {{"destination_type", "i32"}});
        auto Add2 = makeOP<v1::Add>({Divide1, Convert7}, {{"auto_broadcast", "numpy"}});
        auto Concat0 = makeOP<v0::Concat>({Add2, {-1}}, {{"axis", 0}});
        auto VariadicSplit0 = makeOP<v1::VariadicSplit>({Transpose0, -1, Concat0});
        auto Constant12 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        32,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto Unsqueeze1 = makeOP<v0::Unsqueeze>({position_ids, 1});
        auto ShapeOf1 = makeOP<v3::ShapeOf>({Unsqueeze1}, {{"output_type", "i64"}});
        auto Gather2 = makeOP<v8::Gather>({ShapeOf1, {0}, 0}, {{"batch_dims", 0}});
        auto Concat1 = makeOP<v0::Concat>({Gather2, {1l}, {1l}}, {{"axis", 0}});
        auto Broadcast0 = makeOP<v3::Broadcast>({Constant12, Concat1}, {{"mode", "bidirectional"}});
        auto Unsqueeze2 = makeOP<v0::Unsqueeze>({Unsqueeze1, 1});
        auto Convert8 = makeOP<v0::Convert>({Unsqueeze2}, {{"destination_type", "f32"}});
        auto MatMul1 = makeOP<v0::MatMul>({Broadcast0, Convert8}, {{"transpose_a", false}, {"transpose_b", false}});
        auto Transpose1 = makeOP<v1::Transpose>({MatMul1, {0, 2, 1}});
        auto Cos0 = makeOP<v0::Cos>({Transpose1});
        auto Constant13 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        1,
                                        1,
                                    }),
                                    {1.346574f});
        auto Multiply4 = makeOP<v1::Multiply>({Cos0, Constant13}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze3 = makeOP<v0::Unsqueeze>({Multiply4, 1});
        auto Multiply5 = makeOP<v1::Multiply>({VariadicSplit0->output(0), Unsqueeze3}, {{"auto_broadcast", "numpy"}});
        auto Sin0 = makeOP<v0::Sin>({Transpose1});
        auto Constant14 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        1,
                                        1,
                                    }),
                                    {1.346574f});
        auto Multiply6 = makeOP<v1::Multiply>({Sin0, Constant14}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze4 = makeOP<v0::Unsqueeze>({Multiply6, 1});
        auto Multiply7 = makeOP<v1::Multiply>({VariadicSplit0->output(1), Unsqueeze4}, {{"auto_broadcast", "numpy"}});
        auto Subtract2 = makeOP<v1::Subtract>({Multiply5, Multiply7}, {{"auto_broadcast", "numpy"}});
        auto Multiply8 = makeOP<v1::Multiply>({VariadicSplit0->output(1), Unsqueeze3}, {{"auto_broadcast", "numpy"}});
        auto Multiply9 = makeOP<v1::Multiply>({VariadicSplit0->output(0), Unsqueeze4}, {{"auto_broadcast", "numpy"}});
        auto Add3 = makeOP<v1::Add>({Multiply8, Multiply9}, {{"auto_broadcast", "numpy"}});
        auto Concat2 = makeOP<v0::Concat>({Subtract2, Add3}, {{"axis", -1}});
        auto Transpose2 = makeOP<v1::Transpose>({Concat2, {0, 2, 1, 3}});
        auto Reshape1 = makeOP<v1::Reshape>({Transpose2, {0, -1}}, {{"special_zero", true}});
        auto Constant15 = makeConst(element::u8,
                                    ov::Shape({
                                        512,
                                        2880,
                                    }),
                                    MOCK_VALUE);
        auto Convert9 = makeOP<v0::Convert>({Constant15}, {{"destination_type", "f16"}});
        auto Constant16 = makeConst(element::u8,
                                    ov::Shape({
                                        512,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto Convert10 = makeOP<v0::Convert>({Constant16}, {{"destination_type", "f16"}});
        auto Subtract3 = makeOP<v1::Subtract>({Convert9, Convert10}, {{"auto_broadcast", "numpy"}});
        auto Constant17 = makeConst(element::f16,
                                    ov::Shape({
                                        512,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto Multiply10 = makeOP<v1::Multiply>({Subtract3, Constant17}, {{"auto_broadcast", "numpy"}});
        auto Convert11 = makeOP<v0::Convert>({Multiply10}, {{"destination_type", "f32"}});
        auto MatMul2 = makeOP<v0::MatMul>({Multiply2, Convert11}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Reshape2 = makeOP<v1::Reshape>({MatMul2, {0, 0, 8, 64}}, {{"special_zero", true}});
        auto Transpose3 = makeOP<v1::Transpose>({Reshape2, {0, 2, 1, 3}});
        auto ShapeOf2 = makeOP<v3::ShapeOf>({Transpose3}, {{"output_type", "i32"}});
        auto Gather3 = makeOP<v8::Gather>({ShapeOf2, -1, {0}}, {{"batch_dims", 0}});
        auto Divide2 = makeOP<v1::Divide>({Gather3, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Mod1 = makeOP<v1::Mod>({Gather3, 2}, {{"auto_broadcast", "numpy"}});
        auto Greater1 = makeOP<v1::Greater>({Mod1, {0}}, {{"auto_broadcast", "numpy"}});
        auto Convert12 = makeOP<v0::Convert>({Greater1}, {{"destination_type", "i32"}});
        auto Add4 = makeOP<v1::Add>({Divide2, Convert12}, {{"auto_broadcast", "numpy"}});
        auto Concat3 = makeOP<v0::Concat>({Add4, {-1}}, {{"axis", 0}});
        auto VariadicSplit1 = makeOP<v1::VariadicSplit>({Transpose3, -1, Concat3});
        auto Multiply11 = makeOP<v1::Multiply>({VariadicSplit1->output(0), Unsqueeze3}, {{"auto_broadcast", "numpy"}});
        auto Multiply12 = makeOP<v1::Multiply>({VariadicSplit1->output(1), Unsqueeze4}, {{"auto_broadcast", "numpy"}});
        auto Subtract4 = makeOP<v1::Subtract>({Multiply11, Multiply12}, {{"auto_broadcast", "numpy"}});
        auto Multiply13 = makeOP<v1::Multiply>({VariadicSplit1->output(1), Unsqueeze3}, {{"auto_broadcast", "numpy"}});
        auto Multiply14 = makeOP<v1::Multiply>({VariadicSplit1->output(0), Unsqueeze4}, {{"auto_broadcast", "numpy"}});
        auto Add5 = makeOP<v1::Add>({Multiply13, Multiply14}, {{"auto_broadcast", "numpy"}});
        auto Concat4 = makeOP<v0::Concat>({Subtract4, Add5}, {{"axis", -1}});
        auto Transpose4 = makeOP<v1::Transpose>({Concat4, {0, 2, 1, 3}});
        auto Reshape3 = makeOP<v1::Reshape>({Transpose4, {0, -1}}, {{"special_zero", true}});
        auto Constant18 = makeConst(element::u8,
                                    ov::Shape({
                                        512,
                                        2880,
                                    }),
                                    MOCK_VALUE);
        auto Convert13 = makeOP<v0::Convert>({Constant18}, {{"destination_type", "f16"}});
        auto Constant19 = makeConst(element::u8,
                                    ov::Shape({
                                        512,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto Convert14 = makeOP<v0::Convert>({Constant19}, {{"destination_type", "f16"}});
        auto Subtract5 = makeOP<v1::Subtract>({Convert13, Convert14}, {{"auto_broadcast", "numpy"}});
        auto Constant20 = makeConst(element::f16,
                                    ov::Shape({
                                        512,
                                        1,
                                    }),
                                    MOCK_VALUE);
        auto Multiply15 = makeOP<v1::Multiply>({Subtract5, Constant20}, {{"auto_broadcast", "numpy"}});
        auto Convert15 = makeOP<v0::Convert>({Multiply15}, {{"destination_type", "f32"}});
        auto MatMul3 = makeOP<v0::MatMul>({Multiply2, Convert15}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Constant21 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        1,
                                        512,
                                    }),
                                    MOCK_VALUE);
        auto Add6 = makeOP<v1::Add>({MatMul3, Constant21}, {{"auto_broadcast", "numpy"}});
        auto Reshape4 = makeOP<v1::Reshape>({Add6, {0, 0, 8, 64}}, {{"special_zero", true}});
        auto Transpose5 = makeOP<v1::Transpose>({Reshape4, {0, 2, 1, 3}});
        auto Transpose6 = makeOP<v1::Transpose>({Transpose5, {0, 2, 1, 3}});
        auto Reshape5 = makeOP<v1::Reshape>({Transpose6, {0, -1}}, {{"special_zero", true}});
        auto Constant22 = makeConst(element::f32,
                                    ov::Shape({
                                        1,
                                        64,
                                        1,
                                        1,
                                    }),
                                    MOCK_VALUE);

        auto scale = v0::Constant::create(element::f32, {}, {0.125000f});
        auto sliding_window = v0::Constant::create(element::i32, {}, {0});
        auto alibi_slopes_stub = v0::Constant::create(element::f32, Shape{0}, {});
        auto PagedAttentionExtension =
            std::make_shared<ov::op::PagedAttentionExtension>(OutputVector{Reshape1,
                                                                           Reshape3,
                                                                           Reshape5,
                                                                           key_cache_0,
                                                                           value_cache_0,
                                                                           past_lens,
                                                                           subsequence_begins,
                                                                           block_indices,
                                                                           block_indices_begins,
                                                                           scale,
                                                                           sliding_window,
                                                                           alibi_slopes_stub,
                                                                           max_context_len,
                                                                           score_aggregation_window,
                                                                           rotated_block_indices,
                                                                           rotation_deltas,
                                                                           rotation_trig_lut,
                                                                           xattention_threshold,
                                                                           xattention_block_size,
                                                                           xattention_stride,
                                                                           Constant22});
        auto ShapeOf3 = makeOP<v3::ShapeOf>({Transpose6}, {{"output_type", "i64"}});
        auto Gather4 = makeOP<v8::Gather>({ShapeOf3, -1, 0}, {{"batch_dims", 0}});
        auto Unsqueeze5 = makeOP<v0::Unsqueeze>({Gather4, 0});
        auto Concat5 = makeOP<v0::Concat>({{0l}, {1l}, {-1l}, Unsqueeze5}, {{"axis", 0}});
        auto Reshape6 = makeOP<v1::Reshape>({PagedAttentionExtension->output(0), Concat5}, {{"special_zero", true}});
        auto Transpose7 = makeOP<v1::Transpose>({Reshape6, {0, 2, 1, 3}});

        auto res = makeOP<v0::Result>({Transpose7});

        model_ref = std::make_shared<ov::Model>(res, params);

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
