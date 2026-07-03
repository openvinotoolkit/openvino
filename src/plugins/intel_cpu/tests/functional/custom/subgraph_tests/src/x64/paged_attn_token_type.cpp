// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "internal_properties.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;

namespace ov {
namespace test {

struct TokenTypePattern {
    std::string name;
    std::vector<int32_t> types;  // 0=text, 1=image
};

using PagedAttnTokenTypeParams = std::tuple<ElementType, size_t, size_t, TokenTypePattern>;

class PagedAttnTokenTypeTest : public testing::WithParamInterface<PagedAttnTokenTypeParams>,
                               virtual public ov::test::SubgraphBaseTest,
                               public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttnTokenTypeParams>& obj) {
        const auto& [inType, head_size, head_num, pattern] = obj.param;
        std::ostringstream result;
        result << "Prc=" << inType << "_";
        result << "HS=" << head_size << "_";
        result << "HN=" << head_num << "_";
        result << "Pattern=" << pattern.name;
        return result.str();
    }

    static std::shared_ptr<ov::op::v0::Parameter> make_param(const PartialShape& pshape,
                                                             element::Type element_type,
                                                             const std::string& name) {
        auto param = std::make_shared<v0::Parameter>(element_type, pshape);
        param->set_friendly_name(name);
        param->get_output_tensor(0).set_names({name});
        return param;
    }

    std::shared_ptr<ov::Model> get_pa_model(ov::element::Type data_type,
                                            ov::Dimension::value_type head_size,
                                            ov::Dimension::value_type head_num,
                                            int32_t sliding_window_size = 0) {
        auto q = make_param(PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, data_type, "q");
        auto k = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "k");
        auto v = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "v");
        auto key_cache = make_param(PartialShape{ov::Dimension::dynamic(), 32, ov::Dimension::dynamic()},
                                    ov::element::dynamic, "key_cache.0");
        auto value_cache = make_param(PartialShape{ov::Dimension::dynamic(), 32, ov::Dimension::dynamic()},
                                      ov::element::dynamic, "value_cache.0");
        auto past_lens = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "past_lens");
        auto subsequence_begins = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "subsequence_begins");
        auto block_indices = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices");
        auto block_indices_begins = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices_begins");

        float scale_value = 1.0f / std::sqrt(static_cast<float>(head_size));
        auto scale = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale_value});
        auto sliding_window = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{sliding_window_size});
        auto alibi_slopes = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
        auto max_context_len = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{1024});
        // Empty (not a baked scalar) so the model works for batched requests: a
        // per-sequence input sized [1] would be asserted against B_seq and fail
        // for B_seq > 1. Empty means "no score aggregation" and is simply skipped.
        auto score_aggregation_window = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{});
        auto rotated_block_indices = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto rotation_deltas = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto rotation_trig_lut = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
        auto xattention_threshold = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
        auto xattention_block_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{64});
        auto xattention_stride = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{8});
        auto sinks = std::static_pointer_cast<v0::Constant>(
            ov::test::utils::make_constant(data_type, Shape{0}));
        auto adaptive_rkv_start_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        auto adaptive_rkv_evictable_sizes = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto adaptive_rkv_diversity_block_set_indices = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        auto adaptive_rkv_diversity_block_set_indices_begins = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});

        auto token_type_ids = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "token_type_ids");
        auto qq_bias = std::make_shared<v0::Constant>(ov::element::u8, Shape{0}, std::vector<uint8_t>{0});
        auto qq_bias_begins = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
        ParameterVector params = {q, k, v, key_cache, value_cache, past_lens,
                                  subsequence_begins, block_indices, block_indices_begins,
                                  token_type_ids};

        OutputVector pa_inputs = {q, k, v, key_cache, value_cache,
                                  past_lens, subsequence_begins, block_indices, block_indices_begins,
                                  scale, sliding_window, alibi_slopes, max_context_len,
                                  score_aggregation_window, rotated_block_indices, rotation_deltas,
                                  rotation_trig_lut, xattention_threshold, xattention_block_size,
                                  xattention_stride, sinks, adaptive_rkv_start_size,
                                  adaptive_rkv_evictable_sizes,
                                  adaptive_rkv_diversity_block_set_indices,
                                  adaptive_rkv_diversity_block_set_indices_begins,
                                  token_type_ids, qq_bias, qq_bias_begins};

        OPENVINO_ASSERT(pa_inputs.size() == 28);

        auto paged_attn = std::make_shared<op::PagedAttentionExtension>(pa_inputs);
        paged_attn->get_rt_info()["num_k_heads"] = head_num;
        paged_attn->get_rt_info()["k_head_size"] = head_size;
        paged_attn->get_rt_info()["num_v_heads"] = head_num;
        paged_attn->get_rt_info()["v_head_size"] = head_size;

        return std::make_shared<ov::Model>(OutputVector{paged_attn}, params);
    }

    template <typename IT, typename T>
    static void strided_iota(IT first, size_t n, T value, T stride) {
        for (size_t i = 0; i < n; i++) {
            const float idx = static_cast<float>(n - 1 - i);
            *first++ = value + stride * static_cast<T>(idx);
        }
    }

    struct RunResult {
        ov::Tensor output;
    };

    RunResult run_pa_with_token_types(std::shared_ptr<ov::Model> model,
                                     ov::element::Type data_type,
                                     size_t seq_len,
                                     size_t head_size,
                                     size_t head_num,
                                     const std::vector<int32_t>& token_types) {
        OPENVINO_ASSERT(token_types.size() == seq_len);

        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        function = model;
        compile_model();
        auto infer_request = compiledModel.create_infer_request();

        // Determine cache precision from compiled model
        ov::Tensor key_cache_tensor, value_cache_tensor;
        for (const auto& input : compiledModel.inputs()) {
            for (auto& name : input.get_names()) {
                auto cache_precision = input.get_element_type();
                const size_t block_nums = 1024 / 32;
                ov::PartialShape pshape;
                if (name.find("key_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    pshape[0] = block_nums;
                    key_cache_tensor = ov::Tensor(cache_precision, pshape.get_shape());
                } else if (name.find("value_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    pshape[0] = block_nums;
                    value_cache_tensor = ov::Tensor(cache_precision, pshape.get_shape());
                }
            }
        }

        auto params = model->get_parameters();
        size_t hidden_dim = head_num * head_size;

        // q, k, v tensors [seq_len, hidden_dim]
        auto fill_tensor = [](ov::Tensor& t, float base, float stride) {
            auto* p = t.data<float>();
            for (size_t i = 0; i < t.get_size(); i++) {
                p[i] = base + stride * static_cast<float>(i % 17);  // pseudo-random repeating pattern
            }
        };

        ov::Tensor q_tensor(data_type, {seq_len, hidden_dim});
        ov::Tensor k_tensor(data_type, {seq_len, hidden_dim});
        ov::Tensor v_tensor(data_type, {seq_len, hidden_dim});

        if (data_type == ov::element::f32) {
            fill_tensor(q_tensor, 0.1f, 0.01f);
            fill_tensor(k_tensor, 0.2f, 0.01f);
            fill_tensor(v_tensor, 0.3f, 0.01f);
        }

        // Prefill: past_lens=0, single sequence
        size_t batch_size = 1;
        int32_t total_blocks = static_cast<int32_t>((seq_len + 31) / 32);

        ov::Tensor past_lens(ov::element::i32, {batch_size});
        ov::Tensor subsequence_begins(ov::element::i32, {batch_size + 1});
        ov::Tensor block_indices(ov::element::i32, {static_cast<size_t>(total_blocks)});
        ov::Tensor block_indices_begins(ov::element::i32, {batch_size + 1});

        past_lens.data<int32_t>()[0] = 0;
        subsequence_begins.data<int32_t>()[0] = 0;
        subsequence_begins.data<int32_t>()[1] = static_cast<int32_t>(seq_len);
        block_indices_begins.data<int32_t>()[0] = 0;
        block_indices_begins.data<int32_t>()[1] = total_blocks;
        for (int32_t i = 0; i < total_blocks; i++) {
            block_indices.data<int32_t>()[i] = i;
        }

        // token_type_ids
        ov::Tensor token_type_tensor(ov::element::i32, {seq_len});
        std::memcpy(token_type_tensor.data<int32_t>(), token_types.data(), seq_len * sizeof(int32_t));

        for (auto& param : params) {
            auto name = param->get_friendly_name();
            if (name == "q") infer_request.set_tensor(param, q_tensor);
            else if (name == "k") infer_request.set_tensor(param, k_tensor);
            else if (name == "v") infer_request.set_tensor(param, v_tensor);
            else if (name == "key_cache.0") infer_request.set_tensor(param, key_cache_tensor);
            else if (name == "value_cache.0") infer_request.set_tensor(param, value_cache_tensor);
            else if (name == "past_lens") infer_request.set_tensor(param, past_lens);
            else if (name == "subsequence_begins") infer_request.set_tensor(param, subsequence_begins);
            else if (name == "block_indices") infer_request.set_tensor(param, block_indices);
            else if (name == "block_indices_begins") infer_request.set_tensor(param, block_indices_begins);
            else if (name == "token_type_ids") infer_request.set_tensor(param, token_type_tensor);
        }

        infer_request.infer();

        auto output = infer_request.get_output_tensor(0);
        ov::Tensor output_copy{output.get_element_type(), output.get_shape()};
        output.copy_to(output_copy);
        return {output_copy};
    }

    // Process the sequence in two chunks that share a single KV cache (chunked
    // prefill / prefix caching, the second call runs with past_len > 0)
    RunResult run_pa_chunked(std::shared_ptr<ov::Model> model,
                             ov::element::Type data_type,
                             size_t seq_len,
                             size_t head_size,
                             size_t head_num,
                             const std::vector<int32_t>& token_types,
                             size_t chunk1_len) {
        OPENVINO_ASSERT(token_types.size() == seq_len);
        OPENVINO_ASSERT(chunk1_len > 0 && chunk1_len < seq_len);
        OPENVINO_ASSERT(data_type == ov::element::f32, "run_pa_chunked only supports f32 inputs");

        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        function = model;
        compile_model();
        auto infer_request = compiledModel.create_infer_request();

        // KV cache is allocated once and shared across both infer calls so the
        // keys/values written by chunk 1 are visible to chunk 2.
        ov::Tensor key_cache_tensor, value_cache_tensor;
        for (const auto& input : compiledModel.inputs()) {
            for (auto& name : input.get_names()) {
                auto cache_precision = input.get_element_type();
                const size_t block_nums = 1024 / 32;
                ov::PartialShape pshape;
                if (name.find("key_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    pshape[0] = block_nums;
                    key_cache_tensor = ov::Tensor(cache_precision, pshape.get_shape());
                } else if (name.find("value_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    pshape[0] = block_nums;
                    value_cache_tensor = ov::Tensor(cache_precision, pshape.get_shape());
                }
            }
        }
        std::memset(key_cache_tensor.data(), 0, key_cache_tensor.get_byte_size());
        std::memset(value_cache_tensor.data(), 0, value_cache_tensor.get_byte_size());

        auto params = model->get_parameters();
        const size_t hidden_dim = head_num * head_size;

        auto fill_tensor = [](ov::Tensor& t, float base, float stride) {
            auto* p = t.data<float>();
            for (size_t i = 0; i < t.get_size(); i++) {
                p[i] = base + stride * static_cast<float>(i % 17);
            }
        };
        ov::Tensor q_full(data_type, {seq_len, hidden_dim});
        ov::Tensor k_full(data_type, {seq_len, hidden_dim});
        ov::Tensor v_full(data_type, {seq_len, hidden_dim});
        fill_tensor(q_full, 0.1f, 0.01f);
        fill_tensor(k_full, 0.2f, 0.01f);
        fill_tensor(v_full, 0.3f, 0.01f);

        const int32_t total_blocks = static_cast<int32_t>((seq_len + 31) / 32);
        ov::Tensor block_indices(ov::element::i32, {static_cast<size_t>(total_blocks)});
        for (int32_t i = 0; i < total_blocks; i++)
            block_indices.data<int32_t>()[i] = i;

        auto slice_rows = [&](const ov::Tensor& src, size_t row0, size_t nrows) {
            ov::Tensor dst(src.get_element_type(), {nrows, hidden_dim});
            std::memcpy(dst.data<float>(), src.data<float>() + row0 * hidden_dim,
                        nrows * hidden_dim * sizeof(float));
            return dst;
        };

        auto run_chunk = [&](size_t row0, size_t nrows, int32_t past_len, int32_t nblocks) {
            ov::Tensor q = slice_rows(q_full, row0, nrows);
            ov::Tensor k = slice_rows(k_full, row0, nrows);
            ov::Tensor v = slice_rows(v_full, row0, nrows);

            ov::Tensor past_lens(ov::element::i32, {1});
            ov::Tensor subsequence_begins(ov::element::i32, {2});
            ov::Tensor block_indices_begins(ov::element::i32, {2});
            past_lens.data<int32_t>()[0] = past_len;
            subsequence_begins.data<int32_t>()[0] = 0;
            subsequence_begins.data<int32_t>()[1] = static_cast<int32_t>(nrows);
            block_indices_begins.data<int32_t>()[0] = 0;
            block_indices_begins.data<int32_t>()[1] = nblocks;

            ov::Tensor token_type_tensor(ov::element::i32, {nrows});
            std::memcpy(token_type_tensor.data<int32_t>(), token_types.data() + row0,
                        nrows * sizeof(int32_t));

            for (auto& param : params) {
                auto name = param->get_friendly_name();
                if (name == "q") infer_request.set_tensor(param, q);
                else if (name == "k") infer_request.set_tensor(param, k);
                else if (name == "v") infer_request.set_tensor(param, v);
                else if (name == "key_cache.0") infer_request.set_tensor(param, key_cache_tensor);
                else if (name == "value_cache.0") infer_request.set_tensor(param, value_cache_tensor);
                else if (name == "past_lens") infer_request.set_tensor(param, past_lens);
                else if (name == "subsequence_begins") infer_request.set_tensor(param, subsequence_begins);
                else if (name == "block_indices") infer_request.set_tensor(param, block_indices);
                else if (name == "block_indices_begins") infer_request.set_tensor(param, block_indices_begins);
                else if (name == "token_type_ids") infer_request.set_tensor(param, token_type_tensor);
            }

            infer_request.infer();
            auto output = infer_request.get_output_tensor(0);
            ov::Tensor out_copy{output.get_element_type(), output.get_shape()};
            output.copy_to(out_copy);
            return out_copy;
        };

        const int32_t nblocks1 = static_cast<int32_t>((chunk1_len + 31) / 32);
        // Chunk 1: leading tokens, past_len = 0.
        run_chunk(0, chunk1_len, 0, nblocks1);
        // Chunk 2: remaining tokens, past_len = chunk1_len (prefix already cached).
        ov::Tensor chunk2_out = run_chunk(chunk1_len, seq_len - chunk1_len,
                                          static_cast<int32_t>(chunk1_len), total_blocks);
        return {chunk2_out};
    }

    // Batched prefix-caching variant: a leading all-text "filler" sequence (seq A)
    // is placed first so the sequence under test (seq B) starts at a non-zero subsequence_begins
    RunResult run_pa_batched_second_chunked(std::shared_ptr<ov::Model> model,
                                            ov::element::Type data_type,
                                            size_t seqA_len,
                                            const std::vector<int32_t>& seqB_types,
                                            size_t chunkB1_len,
                                            size_t head_size,
                                            size_t head_num) {
        const size_t seqB_len = seqB_types.size();
        OPENVINO_ASSERT(seqA_len > 0);
        OPENVINO_ASSERT(chunkB1_len > 0 && chunkB1_len < seqB_len);
        // Same raw-f32 restriction as run_pa_chunked (see note there).
        OPENVINO_ASSERT(data_type == ov::element::f32, "run_pa_batched_second_chunked only supports f32 inputs");

        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        function = model;
        compile_model();
        auto infer_request = compiledModel.create_infer_request();

        // Single shared KV cache across the priming and the batched infer calls.
        ov::Tensor key_cache_tensor, value_cache_tensor;
        for (const auto& input : compiledModel.inputs()) {
            for (auto& name : input.get_names()) {
                auto cache_precision = input.get_element_type();
                const size_t block_nums = 1024 / 32;
                ov::PartialShape pshape;
                if (name.find("key_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    pshape[0] = block_nums;
                    key_cache_tensor = ov::Tensor(cache_precision, pshape.get_shape());
                } else if (name.find("value_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    pshape[0] = block_nums;
                    value_cache_tensor = ov::Tensor(cache_precision, pshape.get_shape());
                }
            }
        }
        std::memset(key_cache_tensor.data(), 0, key_cache_tensor.get_byte_size());
        std::memset(value_cache_tensor.data(), 0, value_cache_tensor.get_byte_size());

        auto params = model->get_parameters();
        const size_t hidden_dim = head_num * head_size;

        // Fill each sequence's q/k/v with the same per-element pattern the
        // single-shot helper uses, so per-sequence values match exactly.
        auto fill_tensor = [](ov::Tensor& t, float base, float stride) {
            auto* p = t.data<float>();
            for (size_t i = 0; i < t.get_size(); i++) {
                p[i] = base + stride * static_cast<float>(i % 17);
            }
        };
        ov::Tensor qA(data_type, {seqA_len, hidden_dim});
        ov::Tensor kA(data_type, {seqA_len, hidden_dim});
        ov::Tensor vA(data_type, {seqA_len, hidden_dim});
        fill_tensor(qA, 0.1f, 0.01f);
        fill_tensor(kA, 0.2f, 0.01f);
        fill_tensor(vA, 0.3f, 0.01f);
        ov::Tensor qB(data_type, {seqB_len, hidden_dim});
        ov::Tensor kB(data_type, {seqB_len, hidden_dim});
        ov::Tensor vB(data_type, {seqB_len, hidden_dim});
        fill_tensor(qB, 0.1f, 0.01f);
        fill_tensor(kB, 0.2f, 0.01f);
        fill_tensor(vB, 0.3f, 0.01f);

        auto slice_rows = [&](const ov::Tensor& src, size_t row0, size_t nrows) {
            ov::Tensor dst(src.get_element_type(), {nrows, hidden_dim});
            std::memcpy(dst.data<float>(), src.data<float>() + row0 * hidden_dim,
                        nrows * hidden_dim * sizeof(float));
            return dst;
        };

        // Disjoint block layout: seq B owns blocks [0, nblocksB); seq A owns the
        // blocks that immediately follow. The two sequences never share KV storage.
        const int32_t nblocksB = static_cast<int32_t>((seqB_len + 31) / 32);
        const int32_t nblocksA = static_cast<int32_t>((seqA_len + 31) / 32);
        std::vector<int32_t> blocksB(nblocksB), blocksA(nblocksA);
        for (int32_t i = 0; i < nblocksB; i++)
            blocksB[i] = i;
        for (int32_t i = 0; i < nblocksA; i++)
            blocksA[i] = nblocksB + i;

        auto set_inputs = [&](ov::Tensor& q, ov::Tensor& k, ov::Tensor& v, ov::Tensor& past_lens,
                              ov::Tensor& subsequence_begins, ov::Tensor& block_indices,
                              ov::Tensor& block_indices_begins, ov::Tensor& token_type_tensor) {
            for (auto& param : params) {
                auto name = param->get_friendly_name();
                if (name == "q") infer_request.set_tensor(param, q);
                else if (name == "k") infer_request.set_tensor(param, k);
                else if (name == "v") infer_request.set_tensor(param, v);
                else if (name == "key_cache.0") infer_request.set_tensor(param, key_cache_tensor);
                else if (name == "value_cache.0") infer_request.set_tensor(param, value_cache_tensor);
                else if (name == "past_lens") infer_request.set_tensor(param, past_lens);
                else if (name == "subsequence_begins") infer_request.set_tensor(param, subsequence_begins);
                else if (name == "block_indices") infer_request.set_tensor(param, block_indices);
                else if (name == "block_indices_begins") infer_request.set_tensor(param, block_indices_begins);
                else if (name == "token_type_ids") infer_request.set_tensor(param, token_type_tensor);
            }
        };

        // ---- Prime seq B's prefix (chunk 1), past_len = 0, into seq B's blocks ----
        {
            ov::Tensor q = slice_rows(qB, 0, chunkB1_len);
            ov::Tensor k = slice_rows(kB, 0, chunkB1_len);
            ov::Tensor v = slice_rows(vB, 0, chunkB1_len);

            ov::Tensor past_lens(ov::element::i32, {1});
            ov::Tensor subsequence_begins(ov::element::i32, {2});
            ov::Tensor block_indices(ov::element::i32, {static_cast<size_t>(nblocksB)});
            ov::Tensor block_indices_begins(ov::element::i32, {2});
            past_lens.data<int32_t>()[0] = 0;
            subsequence_begins.data<int32_t>()[0] = 0;
            subsequence_begins.data<int32_t>()[1] = static_cast<int32_t>(chunkB1_len);
            block_indices_begins.data<int32_t>()[0] = 0;
            block_indices_begins.data<int32_t>()[1] = nblocksB;
            std::memcpy(block_indices.data<int32_t>(), blocksB.data(), nblocksB * sizeof(int32_t));

            ov::Tensor token_type_tensor(ov::element::i32, {chunkB1_len});
            std::memcpy(token_type_tensor.data<int32_t>(), seqB_types.data(), chunkB1_len * sizeof(int32_t));

            set_inputs(q, k, v, past_lens, subsequence_begins, block_indices, block_indices_begins,
                       token_type_tensor);
            infer_request.infer();
        }

        // ---- Batched call: seq A (fresh prefill) + seq B chunk 2 (past_len = chunkB1_len) ----
        // seq A occupies the leading rows so seq B has seq_begin = seqA_len != 0.
        const size_t chunkB2_len = seqB_len - chunkB1_len;
        const size_t total_tokens = seqA_len + chunkB2_len;

        ov::Tensor q_tensor(data_type, {total_tokens, hidden_dim});
        ov::Tensor k_tensor(data_type, {total_tokens, hidden_dim});
        ov::Tensor v_tensor(data_type, {total_tokens, hidden_dim});
        auto fill_batched = [&](ov::Tensor& dst, const ov::Tensor& srcA, const ov::Tensor& srcB) {
            std::memcpy(dst.data<float>(), srcA.data<float>(), seqA_len * hidden_dim * sizeof(float));
            std::memcpy(dst.data<float>() + seqA_len * hidden_dim,
                        srcB.data<float>() + chunkB1_len * hidden_dim,
                        chunkB2_len * hidden_dim * sizeof(float));
        };
        fill_batched(q_tensor, qA, qB);
        fill_batched(k_tensor, kA, kB);
        fill_batched(v_tensor, vA, vB);

        ov::Tensor past_lens(ov::element::i32, {2});
        ov::Tensor subsequence_begins(ov::element::i32, {3});
        ov::Tensor block_indices(ov::element::i32, {static_cast<size_t>(nblocksA + nblocksB)});
        ov::Tensor block_indices_begins(ov::element::i32, {3});
        past_lens.data<int32_t>()[0] = 0;                                  // seq A: fresh prefill
        past_lens.data<int32_t>()[1] = static_cast<int32_t>(chunkB1_len);  // seq B: prefix cached
        subsequence_begins.data<int32_t>()[0] = 0;
        subsequence_begins.data<int32_t>()[1] = static_cast<int32_t>(seqA_len);
        subsequence_begins.data<int32_t>()[2] = static_cast<int32_t>(total_tokens);
        block_indices_begins.data<int32_t>()[0] = 0;
        block_indices_begins.data<int32_t>()[1] = nblocksA;
        block_indices_begins.data<int32_t>()[2] = nblocksA + nblocksB;
        std::memcpy(block_indices.data<int32_t>(), blocksA.data(), nblocksA * sizeof(int32_t));
        std::memcpy(block_indices.data<int32_t>() + nblocksA, blocksB.data(), nblocksB * sizeof(int32_t));

        // seq A is all-text; seq B chunk-2 token types follow.
        std::vector<int32_t> batched_types(total_tokens, 0);
        std::memcpy(batched_types.data() + seqA_len, seqB_types.data() + chunkB1_len,
                    chunkB2_len * sizeof(int32_t));
        ov::Tensor token_type_tensor(ov::element::i32, {total_tokens});
        std::memcpy(token_type_tensor.data<int32_t>(), batched_types.data(), total_tokens * sizeof(int32_t));

        set_inputs(q_tensor, k_tensor, v_tensor, past_lens, subsequence_begins, block_indices,
                   block_indices_begins, token_type_tensor);
        infer_request.infer();

        auto output = infer_request.get_output_tensor(0);  // [total_tokens, hidden_dim]
        // Extract seq B chunk-2 rows [seqA_len, total_tokens).
        ov::Tensor seqB_out(output.get_element_type(), {chunkB2_len, hidden_dim});
        std::memcpy(seqB_out.data<float>(),
                    output.data<float>() + seqA_len * hidden_dim,
                    chunkB2_len * hidden_dim * sizeof(float));
        return {seqB_out};
    }
};

// With all-zero token_type_ids (text-only), causal masking must hold:
// output at position i depends only on tokens 0..i, so a shorter prefix
// should produce identical outputs for the shared positions.
TEST_P(PagedAttnTokenTypeTest, AllTextIsCausal) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, head_size, head_num, pattern] = this->GetParam();
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    targetDevice = ov::test::utils::DEVICE_CPU;

    size_t full_len = pattern.types.size();
    size_t prefix_len = full_len / 2;  // e.g. 5 out of 10
    OPENVINO_ASSERT(prefix_len > 0);

    std::vector<int32_t> all_text_full(full_len, 0);
    std::vector<int32_t> all_text_prefix(prefix_len, 0);

    auto model_full = get_pa_model(inType, head_size, head_num);
    auto result_full = run_pa_with_token_types(model_full, inType, full_len, head_size, head_num, all_text_full);

    auto model_prefix = get_pa_model(inType, head_size, head_num);
    auto result_prefix = run_pa_with_token_types(model_prefix, inType, prefix_len, head_size, head_num, all_text_prefix);

    // First prefix_len positions of the full run should match the prefix run exactly
    size_t hidden_dim = head_num * head_size;
    ov::Tensor full_prefix_view(result_full.output.get_element_type(),
                                ov::Shape{prefix_len, hidden_dim},
                                result_full.output.data<float>());
    ov::test::utils::compare(full_prefix_view, result_prefix.output, 1e-5);
}


TEST_P(PagedAttnTokenTypeTest, ImageTokensDifferFromCausal) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, head_size, head_num, pattern] = this->GetParam();
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    targetDevice = ov::test::utils::DEVICE_CPU;

    size_t seq_len = pattern.types.size();

    auto pa_model = get_pa_model(inType, head_size, head_num);
    auto result_bidir = run_pa_with_token_types(pa_model, inType, seq_len, head_size, head_num, pattern.types);

    std::vector<int32_t> all_causal(seq_len, 0);
    auto pa_model_causal = get_pa_model(inType, head_size, head_num);
    auto result_causal = run_pa_with_token_types(pa_model_causal, inType, seq_len, head_size, head_num, all_causal);

    // Image tokens should have different output compared to causal-only
    size_t hidden_dim = head_num * head_size;
    auto* bidir_data = result_bidir.output.data<float>();
    auto* causal_data = result_causal.output.data<float>();

    bool any_image_differs = false;
    for (size_t pos = 0; pos < seq_len; pos++) {
        if (pattern.types[pos] != 1) continue;  // Skip text tokens
        // Only image tokens that are NOT the last in their group will differ,
        // because only they gain access to future KV positions.
        for (size_t d = 0; d < hidden_dim; d++) {
            float diff = std::abs(bidir_data[pos * hidden_dim + d] - causal_data[pos * hidden_dim + d]);
            if (diff > 1e-5f) {
                any_image_differs = true;
                break;
            }
        }
        if (any_image_differs) break;
    }
    EXPECT_TRUE(any_image_differs)
        << "Expected image tokens with bidirectional attention to produce different output than causal-only";
}

// Text tokens outside image groups should be unaffected by token_type_ids
TEST_P(PagedAttnTokenTypeTest, TextTokensUnaffected) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, head_size, head_num, pattern] = this->GetParam();
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    targetDevice = ov::test::utils::DEVICE_CPU;

    size_t seq_len = pattern.types.size();

    auto pa_model = get_pa_model(inType, head_size, head_num);
    auto result_bidir = run_pa_with_token_types(pa_model, inType, seq_len, head_size, head_num, pattern.types);

    std::vector<int32_t> all_causal(seq_len, 0);
    auto pa_model_causal = get_pa_model(inType, head_size, head_num);
    auto result_causal = run_pa_with_token_types(pa_model_causal, inType, seq_len, head_size, head_num, all_causal);

    // Text tokens BEFORE the first image group should have identical output
    size_t hidden_dim = head_num * head_size;

    size_t first_image_pos = seq_len;
    for (size_t i = 0; i < seq_len; i++) {
        if (pattern.types[i] == 1) { first_image_pos = i; break; }
    }

    if (first_image_pos > 0) {
        ov::Tensor bidir_text(result_bidir.output.get_element_type(),
                              ov::Shape{first_image_pos, hidden_dim},
                              result_bidir.output.data<float>());
        ov::Tensor causal_text(result_causal.output.get_element_type(),
                               ov::Shape{first_image_pos, hidden_dim},
                               result_causal.output.data<float>());
        ov::test::utils::compare(causal_text, bidir_text, 1e-5);
    }
}


// Text tokens after the last image group should match causal baseline
TEST_P(PagedAttnTokenTypeTest, PostImageTextIsCausal) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, head_size, head_num, pattern] = this->GetParam();
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    targetDevice = ov::test::utils::DEVICE_CPU;

    size_t seq_len = pattern.types.size();

    auto pa_model = get_pa_model(inType, head_size, head_num);
    auto result_bidir = run_pa_with_token_types(pa_model, inType, seq_len, head_size, head_num, pattern.types);

    std::vector<int32_t> all_causal(seq_len, 0);
    auto pa_model_causal = get_pa_model(inType, head_size, head_num);
    auto result_causal = run_pa_with_token_types(pa_model_causal, inType, seq_len, head_size, head_num, all_causal);

    size_t hidden_dim = head_num * head_size;
    auto* bidir_data = result_bidir.output.data<float>();
    auto* causal_data = result_causal.output.data<float>();

    size_t last_image_pos = 0;
    for (size_t i = 0; i < seq_len; i++) {
        if (pattern.types[i] == 1) last_image_pos = i;
    }

    // Text tokens after the last image group should match causal baseline
    for (size_t pos = last_image_pos + 1; pos < seq_len; pos++) {
        ASSERT_EQ(pattern.types[pos], 0) << "Expected text token at position " << pos;
    }
    if (last_image_pos + 1 < seq_len) {
        size_t post_len = seq_len - last_image_pos - 1;
        ov::Tensor bidir_post(result_bidir.output.get_element_type(),
                              ov::Shape{post_len, hidden_dim},
                              bidir_data + (last_image_pos + 1) * hidden_dim);
        ov::Tensor causal_post(result_causal.output.get_element_type(),
                               ov::Shape{post_len, hidden_dim},
                               causal_data + (last_image_pos + 1) * hidden_dim);
        ov::test::utils::compare(causal_post, bidir_post, 1e-5);
    }
}


// Verify that bidirectional image attention interacts correctly with sliding window
TEST_P(PagedAttnTokenTypeTest, ImageTokensWithSlidingWindowDifferFromCausal) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, head_size, head_num, pattern] = this->GetParam();
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    targetDevice = ov::test::utils::DEVICE_CPU;

    const size_t seq_len = pattern.types.size();

    int first_image = -1;
    int last_image = -1;
    for (size_t i = 0; i < seq_len; ++i) {
        if (pattern.types[i] == 1) {
            if (first_image == -1)
                first_image = static_cast<int>(i);
            if (last_image != -1 && pattern.types[last_image] == 1 && static_cast<int>(i) > last_image + 1)
                break;
            last_image = static_cast<int>(i);
        }
    }

    const int group_size = last_image - first_image + 1;
    const int32_t sw = group_size - 1;

    auto model_bidir = get_pa_model(inType, head_size, head_num, sw);
    auto result_bidir = run_pa_with_token_types(model_bidir, inType, seq_len, head_size, head_num, pattern.types);

    std::vector<int32_t> all_text(seq_len, 0);
    auto model_causal = get_pa_model(inType, head_size, head_num, sw);
    auto result_causal = run_pa_with_token_types(model_causal, inType, seq_len, head_size, head_num, all_text);

    const size_t hidden_dim = head_num * head_size;
    const auto* bidir_data  = result_bidir.output.data<float>();
    const auto* causal_data = result_causal.output.data<float>();

    bool any_image_differs = false;
    for (int pos = first_image; pos <= last_image && !any_image_differs; ++pos) {
        for (size_t d = 0; d < hidden_dim; ++d) {
            float diff = std::abs(bidir_data[pos * hidden_dim + d] - causal_data[pos * hidden_dim + d]);
            if (diff > 1e-5f) {
                any_image_differs = true;
                break;
            }
        }
    }
    EXPECT_TRUE(any_image_differs)
        << "Pattern '" << pattern.name << "' with sliding_window=" << sw
        << ": expected image tokens (bidir) to differ from causal baseline, but they were identical.\n"
        << "This indicates the sliding window is incorrectly clipping the image group "
        << "[" << first_image << ", " << last_image << "].\n";
}

TEST_P(PagedAttnTokenTypeTest, ChunkedPrefillMatchesSingleShot) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, head_size, head_num, pattern] = this->GetParam();
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    targetDevice = ov::test::utils::DEVICE_CPU;

    const size_t seq_len = pattern.types.size();

    size_t first_image_pos = seq_len;
    for (size_t i = 0; i < seq_len; ++i) {
        if (pattern.types[i] == 1) { first_image_pos = i; break; }
    }
    ASSERT_GT(first_image_pos, 0u) << "Pattern must start with a text token";
    ASSERT_LT(first_image_pos, seq_len) << "Pattern must contain an image token";
    const size_t chunk1_len = first_image_pos;

    auto model_single = get_pa_model(inType, head_size, head_num);
    auto result_single =
        run_pa_with_token_types(model_single, inType, seq_len, head_size, head_num, pattern.types);

    auto model_chunked = get_pa_model(inType, head_size, head_num);
    auto result_chunked =
        run_pa_chunked(model_chunked, inType, seq_len, head_size, head_num, pattern.types, chunk1_len);

    // Chunk-2 rows [chunk1_len, seq_len) must match the single-shot output
    const size_t hidden_dim = head_num * head_size;
    const size_t chunk2_len = seq_len - chunk1_len;
    ov::Tensor single_tail(result_single.output.get_element_type(),
                           ov::Shape{chunk2_len, hidden_dim},
                           result_single.output.data<float>() + chunk1_len * hidden_dim);
    ov::test::utils::compare(single_tail, result_chunked.output, 1e-4);
}

// Batched-request variant of ChunkedPrefillMatchesSingleShot. A leading all-text
// filler sequence is scheduled first, so the sequence under test is the SECOND
// subsequence and therefore has seq_begin != 0. It is additionally served from a
// shared KV cache (past_len > 0). This exercises the batched-path KV-coordinate
// conversion kv_offset = past_lens[seq] - seq_begin with both terms non-zero,
// which the single-sequence chunked test (seq_begin == 0) does not cover.
TEST_P(PagedAttnTokenTypeTest, ChunkedPrefillBatchedMatchesSingleShot) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, head_size, head_num, pattern] = this->GetParam();
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    targetDevice = ov::test::utils::DEVICE_CPU;

    const size_t seq_len = pattern.types.size();

    size_t first_image_pos = seq_len;
    for (size_t i = 0; i < seq_len; ++i) {
        if (pattern.types[i] == 1) { first_image_pos = i; break; }
    }
    ASSERT_GT(first_image_pos, 0u) << "Pattern must start with a text token";
    ASSERT_LT(first_image_pos, seq_len) << "Pattern must contain an image token";
    const size_t chunk1_len = first_image_pos;

    // Leading filler sequence length; intentionally not a multiple of 32 so the
    // sequence under test starts at a non-block-aligned, non-zero seq_begin.
    const size_t seqA_len = 7;

    // Single-shot reference for the sequence under test.
    auto model_single = get_pa_model(inType, head_size, head_num);
    auto result_single =
        run_pa_with_token_types(model_single, inType, seq_len, head_size, head_num, pattern.types);

    // Batched + chunked run: seq B is second (seq_begin = seqA_len != 0) and its
    // prefix is served from cache (past_len = chunk1_len).
    auto model_batched = get_pa_model(inType, head_size, head_num);
    auto result_batched = run_pa_batched_second_chunked(
        model_batched, inType, seqA_len, pattern.types, chunk1_len, head_size, head_num);

    // Seq B chunk-2 rows [chunk1_len, seq_len) must match the single-shot tail.
    const size_t hidden_dim = head_num * head_size;
    const size_t chunk2_len = seq_len - chunk1_len;
    ov::Tensor single_tail(result_single.output.get_element_type(),
                           ov::Shape{chunk2_len, hidden_dim},
                           result_single.output.data<float>() + chunk1_len * hidden_dim);
    ov::test::utils::compare(single_tail, result_batched.output, 1e-4);
}

namespace {

const std::vector<TokenTypePattern> token_type_patterns = {
    // Symmetric: text + centered image group + text
    {"centered_image", {0, 0, 0, 1, 1, 1, 1, 0, 0, 0}},

    // Image group near the start, more trailing text
    {"early_image", {0, 1, 1, 1, 1, 0, 0, 0, 0, 0}},

    // Image group near the end, more leading text
    {"late_image", {0, 0, 0, 0, 0, 1, 1, 1, 1, 0}},

    // Large image group — almost all image, minimal text framing
    {"large_image", {0, 1, 1, 1, 1, 1, 1, 1, 1, 0}},

    // Two separate image groups with text between and after
    {"two_image_groups", {0, 1, 1, 0, 0, 0, 1, 1, 1, 0}},
};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnTokenType,
                         PagedAttnTokenTypeTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::Values(64),   // head_size
                                            ::testing::Values(8),    // head_num
                                            ::testing::ValuesIn(token_type_patterns)),
                         PagedAttnTokenTypeTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
