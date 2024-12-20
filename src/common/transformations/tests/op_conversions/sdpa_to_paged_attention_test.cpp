// Copyright (C) 2018-2024 Intel Corporation
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

class Qwen7bChatSDPA {
public:
    static std::shared_ptr<Node> gen_embeddings(const std::shared_ptr<Node>& input_ids) {
        auto view_reshape = makeOP<v1::Reshape>({input_ids, {-1, 0}}, {special_zero_true});
        auto input_ids_i64 = makeOP<v0::Convert>({view_reshape}, {dest_type_i64});

        auto weights = makeConst(element::u8, {151936, 4096}, MOCK_VALUE);
        auto weights_fp16 = makeOP<v0::Convert>({weights}, {dest_type_f16});
        auto zero_point = makeConst(element::u8, {151936, 1}, MOCK_VALUE);
        auto zero_point_fp16 = makeOP<v0::Convert>({zero_point}, {dest_type_f16});
        auto zero_point_subtract = makeOP<v1::Subtract>({weights_fp16, zero_point_fp16}, {numpy_broadcast});

        auto scale = makeConst(element::f16, {151936, 1}, MOCK_VALUE);
        auto mul_scale = makeOP<v1::Multiply>({zero_point_subtract, scale}, {numpy_broadcast});
        auto fq_weights = makeOP<v0::Convert>({mul_scale}, {dest_type_f32});

        return makeOP<v8::Gather>({fq_weights, input_ids_i64, 0}, {{"batch_dims", 0}});
    }

    static std::shared_ptr<Node> gen_attention_weights() {
        auto weights = makeConst(element::u8, {12288, 4096}, MOCK_VALUE);
        auto Convert_375820 = makeOP<v0::Convert>({weights}, {dest_type_f16});
        auto attn_c_attn_weight_zero_point = makeConst(element::u8, {12288, 1}, MOCK_VALUE);
        auto Convert_375823 = makeOP<v0::Convert>({attn_c_attn_weight_zero_point}, {dest_type_f16});
        auto attn_c_attn_weight_zero_point_subtract =
            makeOP<v1::Subtract>({Convert_375820, Convert_375823}, {numpy_broadcast});
        auto attn_c_attn_weight_scale = makeConst(element::f16, {12288, 1}, MOCK_VALUE);
        auto attn_c_attn_weight_fq_weights_1 =
            makeOP<v1::Multiply>({attn_c_attn_weight_zero_point_subtract, attn_c_attn_weight_scale}, {numpy_broadcast});
        return makeOP<v0::Convert>({attn_c_attn_weight_fq_weights_1}, {dest_type_f32});
    }

    static std::shared_ptr<Node> gen_qkv_proj(const std::shared_ptr<Node>& embeddings) {
        auto Constant_244726 = single_val(/*rank*/ 3, /*val*/ 1);
        auto Constant_244724 = single_val(/*rank*/ 3, /*val*/ 2);
        auto pow_Power = makeOP<v1::Power>({embeddings, Constant_244724}, {numpy_broadcast});
        auto mean_ReduceMean = makeOP<v1::ReduceMean>({pow_Power, {-1}}, {{"keep_dims", true}});
        auto Constant_244725 = single_val(/*rank*/ 3, /*val*/ 1);
        auto add_Add = makeOP<v1::Add>({mean_ReduceMean, Constant_244725}, {numpy_broadcast});
        auto rsqrt_Sqrt = makeOP<v0::Sqrt>({add_Add});
        auto rsqrt_Divide = makeOP<v1::Divide>({Constant_244726, rsqrt_Sqrt}, {numpy_broadcast, {"m_pythondiv", true}});
        auto mul_Multiply_0 = makeOP<v1::Multiply>({embeddings, rsqrt_Divide}, {numpy_broadcast});
        auto Constant_244727 = makeConst(element::f32, {1, 1, 4096}, MOCK_VALUE);
        auto mul_Multiply_1 = makeOP<v1::Multiply>({mul_Multiply_0, Constant_244727}, {numpy_broadcast});
        auto attention_weights = gen_attention_weights();
        auto linear_MatMul =
            makeOP<v0::MatMul>({mul_Multiply_1, attention_weights}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Constant_244728 = makeConst(element::f32, {1, 1, 12288}, MOCK_VALUE);
        auto linear_add = makeOP<v1::Add>({linear_MatMul, Constant_244728}, {numpy_broadcast});
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
                                          const std::shared_ptr<Node>& sliced_sin_by_current,
                                          const std::shared_ptr<Node>& sliced_cos_by_current) {
        auto current_k = makeOP<v1::Reshape>({qkv_proj->output(idx), {0, 0, 32, 128}}, {special_zero_true});
        auto sliced_k = makeOP<v8::Slice>({current_k, {0}, head_size, {1}, {3}});
        auto mul_Multiply_2 = makeOP<v1::Multiply>({sliced_k, sliced_cos_by_current}, {numpy_broadcast});
        auto reshape_Reshape_1 = makeOP<v1::Reshape>({sliced_k, {0, 0, 32, 2, 64}}, {special_zero_true});
        auto ListUnpack_Split_1 = makeOP<v1::Split>({reshape_Reshape_1, -2}, {{"num_splits", 2}});
        auto ListUnpack_Squeeze_2 = makeOP<v0::Squeeze>({ListUnpack_Split_1->output(1), -2});
        auto Constant_244730 = single_val(/*rank*/ 4, /*val*/ 1);
        auto neg_Multiply_3 = makeOP<v1::Multiply>({ListUnpack_Squeeze_2, Constant_244730}, {numpy_broadcast});
        auto ListUnpack_Squeeze_1 = makeOP<v0::Squeeze>({ListUnpack_Split_1->output(0), -2});
        auto cat_Concat_2 = makeOP<v0::Concat>({neg_Multiply_3, ListUnpack_Squeeze_1}, {{"axis", -1}});
        auto mul_Multiply_3 = makeOP<v1::Multiply>({cat_Concat_2, sliced_sin_by_current}, {numpy_broadcast});
        return makeOP<v1::Add>({mul_Multiply_2, mul_Multiply_3}, {numpy_broadcast});
    }

    static std::shared_ptr<Node> gen_rope_emb_sin(const std::shared_ptr<Node>& total_seq_len,
                                                  const std::shared_ptr<Node>& neg_mul,
                                                  std::shared_ptr<Node>& head_size) {
        auto sin = makeConst(element::f32, {1, 4096, 1, 128}, MOCK_VALUE);
        auto sliced_sin_by_total = makeOP<v8::Slice>({sin, {0}, total_seq_len, {1}, {1}});
        auto rotery_emb_sin_shape = makeOP<v3::ShapeOf>({sliced_sin_by_total}, {{"output_type", "i64"}});
        head_size = makeOP<v8::Gather>({rotery_emb_sin_shape, {3}, 0}, {{"batch_dims", 0}});
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
        auto slice_Slice_10 = makeConst(element::f32, {1, 32767, 1, 1}, MOCK_VALUE);
        auto slice_Slice_13 = makeOP<v8::Slice>({slice_Slice_10, past_seq_len_2, total_seq_len_2, {1}, {1}});
        auto mul_Multiply_4 = makeOP<v1::Multiply>({rope_Q, slice_Slice_13}, {numpy_broadcast});
        return makeOP<v1::Transpose>({mul_Multiply_4, {0, 2, 1, 3}});
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
        auto slice_Slice_17 = makeConst(element::boolean, {1, 1, 8192, 8192}, MOCK_VALUE);
        auto shape_of_q = makeOP<v3::ShapeOf>({Q_in}, {{"output_type", "i64"}});
        auto Gather_255227 = makeOP<v8::Gather>({shape_of_q, {2}, 0}, {{"batch_dims", 0}});
        auto sub_Subtract_1 = makeOP<v1::Subtract>({total_seq_len, Gather_255227}, {numpy_broadcast});
        auto Concat_238310 = makeOP<v0::Concat>({sub_Subtract_1, {0ll}}, {{"axis", 0}});
        auto Concat_238311 = makeOP<v3::Broadcast>({total_seq_len, {2}}, {{"mode", "numpy"}});
        auto slice_Slice_19 = makeOP<v8::Slice>({slice_Slice_17, Concat_238310, Concat_238311, {1, 1}, {2, 3}});
        auto bitwise_not_BitwiseNot = makeOP<v13::BitwiseNot>({slice_Slice_19});
        auto Constant_244732 = single_val(/*rank*/ 4, /*val*/ 1);
        auto view_Reshape_3 = makeOP<v1::Reshape>({attention_mask_in, {0, 0}}, {special_zero_true});
        auto unsqueeze_Unsqueeze = makeOP<v0::Unsqueeze>({view_Reshape_3, 1});
        auto unsqueeze_Unsqueeze_1 = makeOP<v0::Unsqueeze>({unsqueeze_Unsqueeze, 2});
        auto to_Convert = makeOP<v0::Convert>({unsqueeze_Unsqueeze_1}, {dest_type_f32});
        auto Constant_244731 = single_val(/*rank*/ 4, /*val*/ 1);
        auto rsub_Multiply = makeOP<v1::Multiply>({to_Convert, Constant_244731}, {numpy_broadcast});
        auto rsub_Subtract = makeOP<v1::Subtract>({Constant_244732, rsub_Multiply}, {numpy_broadcast});
        auto Constant_244733 = single_val(/*rank*/ 4, /*val*/ 1);
        auto mul_Multiply_5 = makeOP<v1::Multiply>({rsub_Subtract, Constant_244733}, {numpy_broadcast});
        auto ListConstruct_5 = makeOP<v0::Concat>({{1ll}, {1ll}, Gather_255227, {1ll}}, {{"axis", 0}});
        auto expand_Broadcast = makeOP<v3::Broadcast>({mul_Multiply_5, ListConstruct_5}, {{"mode", "bidirectional"}});
        return makeOP<v1::Select>({bitwise_not_BitwiseNot, -FLT_MAX, expand_Broadcast}, {numpy_broadcast});
    }
};

class Qwen7bChatPA {
public:
    static std::shared_ptr<Node> gen_embeddings(const std::shared_ptr<Node>& input_ids) {
        auto Constant_241 = makeConst(element::u8, {151936, 4096}, MOCK_VALUE);
        auto Convert_242 = makeOP<v0::Convert>({Constant_241}, {dest_type_f16});
        auto Constant_243 = makeConst(element::u8, {151936, 1}, MOCK_VALUE);
        auto Convert_244 = makeOP<v0::Convert>({Constant_243}, {dest_type_f16});
        auto Subtract_245 = makeOP<v1::Subtract>({Convert_242, Convert_244}, {numpy_broadcast});
        auto Constant_246 = makeConst(element::f16, {151936, 1}, MOCK_VALUE);
        auto Multiply_247 = makeOP<v1::Multiply>({Subtract_245, Constant_246}, {numpy_broadcast});
        auto Convert_248 = makeOP<v0::Convert>({Multiply_247}, {dest_type_f32});
        auto Reshape_239 = makeOP<v1::Reshape>({input_ids, {-1, 0}}, {special_zero_true});
        auto Convert_240 = makeOP<v0::Convert>({Reshape_239}, {dest_type_i64});
        return makeOP<v8::Gather>({Convert_248, Convert_240, 0}, {{"batch_dims", 0}});
    }

    static std::shared_ptr<Node> gen_qkv_proj(const std::shared_ptr<Node>& embeddings) {
        auto Constant_236 = makeConst(element::f32, {1, 1, 1}, MOCK_VALUE);
        auto Constant_237 = makeConst(element::f32, {1, 1, 1}, MOCK_VALUE);
        auto Power_251 = makeOP<v1::Power>({embeddings, Constant_237}, {numpy_broadcast});
        auto ReduceMean_253 = makeOP<v1::ReduceMean>({Power_251, {-1}}, {{"keep_dims", true}});
        auto Constant_254 = makeConst(element::f32, {1, 1, 1}, MOCK_VALUE);
        auto Add_255 = makeOP<v1::Add>({ReduceMean_253, Constant_254}, {numpy_broadcast});
        auto Sqrt_256 = makeOP<v0::Sqrt>({Add_255});
        auto Divide_257 = makeOP<v1::Divide>({Constant_236, Sqrt_256}, {numpy_broadcast, {"m_pythondiv", true}});
        auto Multiply_258 = makeOP<v1::Multiply>({embeddings, Divide_257}, {numpy_broadcast});
        auto Constant_259 = makeConst(element::f32, {1, 1, 4096}, MOCK_VALUE);
        auto Multiply_260 = makeOP<v1::Multiply>({Multiply_258, Constant_259}, {numpy_broadcast});
        auto Constant_261 = makeConst(element::u8, {12288, 4096}, MOCK_VALUE);
        auto Convert_262 = makeOP<v0::Convert>({Constant_261}, {dest_type_f16});
        auto Constant_263 = makeConst(element::u8, {12288, 1}, MOCK_VALUE);
        auto Convert_264 = makeOP<v0::Convert>({Constant_263}, {dest_type_f16});
        auto Subtract_265 = makeOP<v1::Subtract>({Convert_262, Convert_264}, {numpy_broadcast});
        auto Constant_266 = makeConst(element::f16, {12288, 1}, MOCK_VALUE);
        auto Multiply_267 = makeOP<v1::Multiply>({Subtract_265, Constant_266}, {numpy_broadcast});
        auto Convert_268 = makeOP<v0::Convert>({Multiply_267}, {dest_type_f32});
        auto MatMul_269 =
            makeOP<v0::MatMul>({Multiply_260, Convert_268}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Constant_270 = makeConst(element::f32, {1, 1, 12288}, MOCK_VALUE);
        auto Add_271 = makeOP<v1::Add>({MatMul_269, Constant_270}, {numpy_broadcast});

        return makeOP<v1::VariadicSplit>({Add_271, 2, {4096, 4096, -1}});
    }

    static std::shared_ptr<Node> gen_rope(QKV idx,
                                          const std::shared_ptr<Node>& qkv_proj,
                                          const std::shared_ptr<Node>& head_size,
                                          const std::shared_ptr<Node>& sliced_sin_by_current,
                                          const std::shared_ptr<Node>& sliced_cos_by_current) {
        auto Reshape_276 = makeOP<v1::Reshape>({qkv_proj->output(idx), {0, 0, 32, 128}}, {special_zero_true});
        auto Slice_437 = makeOP<v8::Slice>({Reshape_276, {0}, head_size, {1}, {3}});
        auto Multiply_440 = makeOP<v1::Multiply>({Slice_437, sliced_cos_by_current}, {numpy_broadcast});
        auto Reshape_442 = makeOP<v1::Reshape>({Slice_437, {0, 0, 32, 2, 64}}, {special_zero_true});
        auto Split_444 = makeOP<v1::Split>({Reshape_442, -2}, {{"num_splits", 2}});
        auto Squeeze_446 = makeOP<v0::Squeeze>({Split_444->output(1), -2});
        auto Constant_447 = makeConst(element::f32, {1, 1, 1, 1}, {1.000000f});
        auto Multiply_448 = makeOP<v1::Multiply>({Squeeze_446, Constant_447}, {numpy_broadcast});
        auto Squeeze_450 = makeOP<v0::Squeeze>({Split_444->output(0), -2});
        auto Concat_451 = makeOP<v0::Concat>({Multiply_448, Squeeze_450}, {{"axis", -1}});
        auto Multiply_461 = makeOP<v1::Multiply>({Concat_451, sliced_sin_by_current}, {numpy_broadcast});
        return makeOP<v1::Add>({Multiply_440, Multiply_461}, {numpy_broadcast});
    }

    static std::shared_ptr<Node> gen_rope_emb_sin(const std::shared_ptr<Node>& max_context_len,
                                                  const std::shared_ptr<Node>& position_ids,
                                                  std::shared_ptr<Node>& head_size) {
        auto Constant_277 = makeConst(element::f32, {1, 4096, 1, 128}, MOCK_VALUE);
        auto Slice_293 = makeOP<v8::Slice>({Constant_277, {0}, max_context_len, {1}, {1}});
        auto slice_sin = makeOP<v8::Gather>({Slice_293, position_ids, 1}, {{"batch_dims", 0}});
        auto ShapeOf_430 = makeOP<opset3::ShapeOf>({Slice_293}, {{"output_type", "i64"}});
        head_size = makeOP<v8::Gather>({ShapeOf_430, {3}, 0}, {{"batch_dims", 0}});
        return makeOP<v1::Reshape>({slice_sin, {-1, 1, 1, 128}}, {{"special_zero", false}});
    }

    static std::shared_ptr<Node> gen_rope_emb_cos(const std::shared_ptr<Node>& max_context_len,
                                                  const std::shared_ptr<Node>& position_ids) {
        auto Constant_452 = makeConst(element::f32, {1, 4096, 1, 128}, MOCK_VALUE);
        auto Slice_456 = makeOP<v8::Slice>({Constant_452, {0}, max_context_len, {1}, {1}});
        auto Slice_460 = makeOP<v8::Gather>({Slice_456, position_ids, 1}, {{"batch_dims", 0}});
        return makeOP<v1::Reshape>({Slice_460, {-1, 1, 1, 128}}, {{"special_zero", false}});
    }

    static std::shared_ptr<Node> align_pa_layout(const std::shared_ptr<Node>& pa,
                                                 const std::shared_ptr<Node>& head_size) {
        auto Concat_1257 = makeOP<v0::Concat>({{0ll}, {1ll}, {-1ll}, head_size}, {{"axis", 0}});
        auto Reshape_1258 = makeOP<v1::Reshape>({pa->output(0), Concat_1257}, {special_zero_true});
        return makeOP<v1::Transpose>({Reshape_1258, {0, 2, 1, 3}});
    }

    static std::shared_ptr<Node> gen_current_len(const std::shared_ptr<Node>& rope_K) {
        auto ShapeOf_484 = makeOP<opset3::ShapeOf>({rope_K}, {{"output_type", "i32"}});
        return makeOP<v8::Gather>({ShapeOf_484, {1}, 0ll}, {{"batch_dims", 0}});
    }

    static std::shared_ptr<Node> gen_past_len(const std::shared_ptr<Node>& input_ids,
                                              const std::shared_ptr<Node>& max_context_len) {
        auto ShapeOf_897 = makeOP<opset3::ShapeOf>({input_ids}, {{"output_type", "i64"}});
        auto Gather_898 = makeOP<v8::Gather>({ShapeOf_897, 1ll, 0ll}, {{"batch_dims", 0}});
        auto Convert_899 = makeOP<v0::Convert>({Gather_898}, {{"destination_type", "i32"}});
        auto past_len = makeOP<v1::Subtract>({max_context_len, Convert_899}, {numpy_broadcast});
        auto Convert_1000 = makeOP<v0::Convert>({past_len}, {{"destination_type", "i32"}});
        return makeOP<v1::Reshape>({Convert_1000, {1}}, {special_zero_true});
    }

    static std::shared_ptr<Node> gen_total_len(const std::shared_ptr<Node>& cur_len,
                                               const std::shared_ptr<Node>& past_len) {
        return makeOP<v1::Add>({past_len, cur_len}, {numpy_broadcast});
    }

    static std::shared_ptr<Node> gen_V(const std::shared_ptr<Node>& qkv_proj, std::shared_ptr<Node>& head_size) {
        auto Reshape_641 = makeOP<v1::Reshape>({qkv_proj->output(2), {0, 0, 32, 128}}, {special_zero_true});
        auto Gather_1231 = makeOP<v8::Gather>({{0, 2, 1, 3}, {0, 2, 1, 3}, 0ll}, {{"batch_dims", 0}});
        auto Transpose_1232 = makeOP<v1::Transpose>({Reshape_641, Gather_1231});

        auto ShapeOf_1250 = makeOP<opset3::ShapeOf>({Transpose_1232}, {{"output_type", "i64"}});
        auto Gather_1251 = makeOP<v8::Gather>({ShapeOf_1250, -1ll, 0ll}, {{"batch_dims", 0}});
        head_size = makeOP<v0::Unsqueeze>({Gather_1251, 0});

        return makeOP<v1::Reshape>({Transpose_1232, {0, -1}}, {special_zero_true});
    }

    static std::shared_ptr<Node> gen_K(const std::shared_ptr<Node>& rope_K) {
        auto Gather_1227 = makeOP<v8::Gather>({{0, 2, 1, 3}, {0, 2, 1, 3}, 0ll}, {{"batch_dims", 0}});
        auto Transpose_1228 = makeOP<v1::Transpose>({rope_K, Gather_1227});
        return makeOP<v1::Reshape>({Transpose_1228, {0, -1}}, {special_zero_true});
    }

    static std::shared_ptr<Node> gen_Q(const std::shared_ptr<Node>& total_seq_len,
                                       const std::shared_ptr<Node>& rope_Q) {
        auto Constant_463 = makeConst(element::f32, {1, 32767, 1, 1}, MOCK_VALUE);
        auto ShapeOf_489 = makeOP<opset3::ShapeOf>({rope_Q}, {{"output_type", "i32"}});
        auto Gather_492 = makeOP<v8::Gather>({ShapeOf_489, {1}, 0ll}, {{"batch_dims", 0}});
        auto past_seq_len_2 = makeOP<v1::Subtract>({total_seq_len, Gather_492}, {numpy_broadcast});
        auto Slice_496 = makeOP<v8::Slice>({Constant_463, past_seq_len_2, total_seq_len, {1}, {1}});
        auto Multiply_631 = makeOP<v1::Multiply>({rope_Q, Slice_496}, {numpy_broadcast});
        auto Transpose_633 = makeOP<v1::Transpose>({Multiply_631, {0, 2, 1, 3}});

        auto Transpose_1223 = makeOP<v1::Transpose>({Transpose_633, {0, 2, 1, 3}});
        return makeOP<v1::Reshape>({Transpose_1223, {0, -1}}, {special_zero_true});
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
        auto Convert_1001 = makeOP<v0::Convert>({max_context_len}, {dest_type_i64});
        auto max_context_len_aligned = makeOP<v1::Reshape>({Convert_1001, {1}}, {special_zero_true});
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
