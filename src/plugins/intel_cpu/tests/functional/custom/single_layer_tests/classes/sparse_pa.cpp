// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_pa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils/precision_support.h"
#include "utils/qsa_indexer_reference.hpp"
#include "utils/quantization_utils.hpp"

namespace {

// Synthetic indexer output for one (token, selection head): every complete block, including one ending exactly at the
// token, may be selected; entries after the count are -1.
void make_selection(ov::test::SparsePASelection mode,
                    bool shuffle,
                    size_t max_blocks,
                    size_t num_complete,
                    std::mt19937& rng,
                    int32_t* indices,
                    int32_t& count) {
    std::fill(indices, indices + max_blocks, -1);
    count = 0;
    if (mode == ov::test::SparsePASelection::kEmpty || num_complete == 0) {
        return;
    }
    const auto below = [&rng](size_t n) {
        return std::uniform_int_distribution<size_t>(0, n - 1)(rng);
    };
    const size_t limit = std::min(num_complete, max_blocks);
    size_t n = limit;
    if (mode == ov::test::SparsePASelection::kRecent) {
        for (size_t k = 0; k < n; k++) {
            indices[k] = static_cast<int32_t>(num_complete - n + k);
        }
    } else {
        n = below(limit + 1);
        std::vector<int32_t> pool(num_complete);
        std::iota(pool.begin(), pool.end(), 0);
        for (size_t k = 0; k < n; k++) {
            const size_t j = k + below(num_complete - k);
            std::swap(pool[k], pool[j]);
        }
        std::sort(pool.begin(), pool.begin() + static_cast<ptrdiff_t>(n));
        std::copy_n(pool.begin(), n, indices);
    }
    // the order of the blocks of a row is up to the producer: ascending, like the CPU indexer returns them, or random
    if (shuffle) {
        std::shuffle(indices, indices + n, rng);
    }
    count = static_cast<int32_t>(n);
}

// Quantize + dequantize one by-token group like the plugin: fma(x, 1 / scale, zp) rounded to nearest even (x / scale +
// zp with std::round differs near ties).
void fake_quant_group(bool u4, const float* src, float* dst, size_t n) {
    float max_val = -std::numeric_limits<float>::max();
    float min_val = std::numeric_limits<float>::max();
    for (size_t i = 0; i < n; i++) {
        max_val = std::max(max_val, src[i]);
        min_val = std::min(min_val, src[i]);
    }
    const float levels = u4 ? 15.0F : 255.0F;
    float scale = (max_val - min_val) / levels;
    if (scale == 0.0F) {
        scale = 0.0001F;
    }
    const float zp = -min_val / scale;
    const float inv_scale = 1.0F / scale;
    for (size_t i = 0; i < n; i++) {
        const float q = std::min(levels, std::max(0.0F, std::nearbyint(std::fma(src[i], inv_scale, zp))));
        dst[i] = (q - zp) * scale;
    }
}

void fake_quant_by_token(ov::element::Type prec, size_t group_size, const float* src, float* dst, size_t head) {
    const size_t group = group_size ? group_size : head;
    for (size_t g = 0; g < head; g += group) {
        fake_quant_group(prec == ov::element::u4, src + g, dst + g, group);
    }
}

// Per-channel scale and zero point over the rows of a key page, as the plugin's quantize_by_channel().
void fake_quant_by_channel(ov::element::Type prec, const float* src, float* dst, size_t rows, size_t hidden) {
    const float levels = prec == ov::element::u4 ? 15.0F : 255.0F;
    for (size_t j = 0; j < hidden; j++) {
        float max_val = -std::numeric_limits<float>::max();
        float min_val = std::numeric_limits<float>::max();
        for (size_t i = 0; i < rows; i++) {
            max_val = std::max(max_val, src[i * hidden + j]);
            min_val = std::min(min_val, src[i * hidden + j]);
        }
        float scale = (max_val - min_val) * (1.0F / levels);
        if (scale == 0.0F) {
            scale = 0.0001F;
        }
        const float zp = -min_val / scale;
        const float inv_scale = 1.0F / scale;
        for (size_t i = 0; i < rows; i++) {
            const float q =
                std::min(levels, std::max(0.0F, std::nearbyint(std::fma(src[i * hidden + j], inv_scale, zp))));
            dst[i * hidden + j] = (q - zp) * scale;
        }
    }
}

struct ReferenceDims {
    size_t num_heads;
    size_t num_kv_heads;
    size_t num_selection_heads;
    size_t k_head_size;
    size_t v_head_size;
    size_t key_block_size;
    size_t max_selected_blocks;
};

std::vector<float> run_sparse_pa_reference(const ReferenceDims& d,
                                           size_t chunk_len,
                                           float scale,
                                           int64_t past_len,
                                           const std::vector<float>& query,
                                           const std::vector<float>& key,
                                           const std::vector<float>& value,
                                           const std::vector<int32_t>& sel_indices,
                                           const std::vector<int32_t>& sel_count) {
    const int64_t history_len = past_len + static_cast<int64_t>(chunk_len);
    auto key_at = [&](size_t head, int64_t pos) {
        return &key[(head * static_cast<size_t>(history_len) + static_cast<size_t>(pos)) * d.k_head_size];
    };
    auto value_at = [&](size_t head, int64_t pos) {
        return &value[(head * static_cast<size_t>(history_len) + static_cast<size_t>(pos)) * d.v_head_size];
    };

    std::vector<float> output(chunk_len * d.num_heads * d.v_head_size, 0.0F);
    const size_t group = d.num_heads / d.num_kv_heads;
    const size_t q_per_sel = d.num_heads / d.num_selection_heads;
    const auto Bk = static_cast<int64_t>(d.key_block_size);
    for (size_t l = 0; l < chunk_len; l++) {
        const int64_t token_pos = past_len + static_cast<int64_t>(l);
        for (size_t h = 0; h < d.num_heads; h++) {
            const size_t kv_head = h / group;
            const size_t sh = h / q_per_sel;
            const float* q = &query[(l * d.num_heads + h) * d.k_head_size];
            std::vector<float> acc(d.v_head_size, 0.0F);
            float running_max = -std::numeric_limits<float>::infinity();
            float running_sum = 0.0F;
            // One online-softmax step for key/value position `kpos`: score = (q . K) * scale, rescale the running
            // max/sum by corr, then acc = acc * corr + p * V. Positions outside the valid history are skipped.
            auto attend_token = [&](int64_t kpos) {
                if (kpos < 0 || kpos >= history_len || kpos > token_pos) {
                    return;
                }
                const float* k = key_at(kv_head, kpos);
                const float* v = value_at(kv_head, kpos);
                float dot = 0.0F;
                for (size_t i = 0; i < d.k_head_size; i++) {
                    dot += q[i] * k[i];
                }
                const float score = dot * scale;
                const float new_max = std::max(running_max, score);
                const float corr = std::exp(running_max - new_max);
                const float p = std::exp(score - new_max);
                running_sum = running_sum * corr + p;
                for (size_t i = 0; i < d.v_head_size; i++) {
                    acc[i] = acc[i] * corr + p * v[i];
                }
                running_max = new_max;
            };

            const size_t row = l * d.num_selection_heads + sh;
            const auto nblocks = static_cast<size_t>(sel_count[row]);
            for (size_t i = 0; i < nblocks; i++) {
                const int64_t block_start = static_cast<int64_t>(sel_indices[row * d.max_selected_blocks + i]) * Bk;
                for (int64_t p = block_start; p < block_start + Bk; p++) {
                    attend_token(p);
                }
            }

            // sel_indices/sel_count only ever carry *complete* blocks. The consumer derives and attends the
            // incomplete block of the query itself, which is empty when the position ends a block.
            for (int64_t p = ((token_pos + 1) / Bk) * Bk; p <= token_pos; p++) {
                attend_token(p);
            }

            const float inv_sum = running_sum > 0.0F ? 1.0F / running_sum : 0.0F;
            for (size_t i = 0; i < d.v_head_size; i++) {
                output[(l * d.num_heads + h) * d.v_head_size + i] = acc[i] * inv_sum;
            }
        }
    }
    return output;
}

const char* invalid_name(ov::test::SparsePAInvalid invalid) {
    switch (invalid) {
    case ov::test::SparsePAInvalid::kValueCachePageSize:
        return "ValueCachePageSize";
    case ov::test::SparsePAInvalid::kValueCacheHeadSize:
        return "ValueCacheHeadSize";
    case ov::test::SparsePAInvalid::kKvHeadCount:
        return "KvHeadCount";
    case ov::test::SparsePAInvalid::kBlockIdOutOfRange:
        return "BlockIdOutOfRange";
    case ov::test::SparsePAInvalid::kNegativeBlockId:
        return "NegativeBlockId";
    case ov::test::SparsePAInvalid::kSubsequenceBeginsShort:
        return "SubsequenceBeginsShort";
    default:
        return "None";
    }
}

const char* selection_name(ov::test::SparsePASelection selection) {
    switch (selection) {
    case ov::test::SparsePASelection::kRecent:
        return "Recent";
    case ov::test::SparsePASelection::kRandom:
        return "Random";
    default:
        return "Empty";
    }
}

}  // namespace

namespace ov {
namespace test {

std::string SparsePALayerCPUTest::getTestCaseName(const testing::TestParamInfo<SparsePALayerTestParams>& obj) {
    const auto& [data_precision, shapes, cfg, inference_precision] = obj.param;
    std::ostringstream result;
    result << "Prc=" << data_precision << "_Hint=" << inference_precision << "_Steps=";
    for (const auto& step : shapes.at(0).second) {
        result << step[0] << "_";
    }
    const auto& first = shapes.at(0).second.at(0);
    result << "B=" << first[1] << "_Hq=" << first[2] << "_Dh=" << first[3] << "_Hkv=" << cfg.num_kv_heads
           << "_Hsel=" << cfg.num_selection_heads << "_Dv=" << cfg.v_head_size << "_Bk=" << cfg.key_block_size
           << "_Page=" << cfg.page_size << "_Kmax=" << cfg.max_selected_blocks
           << "_Sel=" << selection_name(cfg.selection) << (cfg.shuffle_blocks ? "_Shuffled" : "")
           << "_Cache=" << (cfg.cache_precision == ov::element::dynamic ? "plain" : cfg.cache_precision.get_type_name())
           << "_KeyByChannel=" << cfg.key_bychannel << "_Group=" << cfg.group_size << "_Mixed=" << cfg.mixed_lengths;
    if (cfg.invalid != SparsePAInvalid::kNone) {
        result << "_Invalid=" << invalid_name(cfg.invalid);
    }
    return result.str();
}

float SparsePALayerCPUTest::through_cache(float value) const {
    if (cache_type() == ov::element::f16) {
        return static_cast<float>(ov::float16(value));
    }
    return value;
}

size_t SparsePALayerCPUTest::sequence_len(size_t step_len, size_t sequence) const {
    if (!m_cfg.mixed_lengths) {
        return step_len;
    }
    switch (sequence % 3) {
    case 0:
        return step_len;  // (chunked) prefill
    case 1:
        return 1;  // decode
    default:
        return (step_len + 1) / 2;  // a shorter chunk
    }
}

ov::Shape SparsePALayerCPUTest::cache_shape(bool is_key) const {
    const size_t head = is_key ? m_k_head_size : m_cfg.v_head_size;
    const size_t n_phys = m_num_sequences * m_pages_per_sequence;
    if (!quantized()) {
        return {n_phys, m_cfg.num_kv_heads, m_cfg.page_size, head};
    }
    // Scale and zero point are f32; u4 counts them in half-byte elements.
    const size_t params = m_cfg.cache_precision == ov::element::u4 ? 16 : 8;
    const size_t group = m_cfg.group_size ? m_cfg.group_size : head;
    if (is_key && m_cfg.key_bychannel) {
        return {n_phys, m_cfg.num_kv_heads, m_cfg.page_size + params, head};
    }
    return {n_phys, m_cfg.num_kv_heads, m_cfg.page_size, head + params * (head / group)};
}

void SparsePALayerCPUTest::SetUp() {
    const auto& [data_precision, shapes, cfg, inference_precision] = GetParam();

    if (!ov::intel_cpu::hasHardwareSupport(data_precision)) {
        GTEST_SKIP() << "CPU precision policy does not preserve " << data_precision << " data on this system.";
    }
    m_inference_precision = inference_precision == ov::element::dynamic ? data_precision : inference_precision;
    if (!ov::intel_cpu::hasHardwareSupport(m_inference_precision)) {
        GTEST_SKIP() << "CPU precision policy does not preserve " << m_inference_precision << " on this system.";
    }
    targetDevice = ov::test::utils::DEVICE_CPU;
    m_data_precision = data_precision;
    m_cfg = cfg;
    m_steps = shapes.at(0).second;
    ASSERT_FALSE(m_steps.empty());
    m_num_sequences = m_steps[0][1];
    m_num_heads = m_steps[0][2];
    m_k_head_size = m_steps[0][3];
    // Tokens every sequence holds after all the steps; the caches have room for the longest one.
    std::vector<size_t> sequence_tokens(m_num_sequences, 0);
    for (const auto& step : m_steps) {
        ASSERT_EQ(step.size(), 4U) << "SparsePA test expects [L, B, H, S] step shapes";
        ASSERT_EQ(step[1], m_num_sequences) << "the number of sequences is fixed across the steps";
        ASSERT_EQ(step[2], m_num_heads);
        ASSERT_EQ(step[3], m_k_head_size);
        for (size_t s = 0; s < m_num_sequences; s++) {
            sequence_tokens[s] += sequence_len(step[0], s);
        }
    }
    const size_t max_tokens = *std::max_element(sequence_tokens.begin(), sequence_tokens.end());
    m_pages_per_sequence = (max_tokens + m_cfg.page_size - 1) / m_cfg.page_size;
    m_total_len.assign(m_num_sequences, 0);
    m_scale = 1.0F / std::sqrt(static_cast<float>(m_k_head_size));

    rel_threshold = 1e-3F;
    abs_threshold = 1e-4F;
    configuration.insert(ov::hint::inference_precision(m_inference_precision));
    // A prefill runs as AMX products at any thread count; a decode packs heads only with enough work items (mixed: 2
    // threads). An invalid case needs 2 threads: with 1 thread and 1 stream the plugin drops a node's exception
    // (CPUStreamsExecutor::execute() catches it).
    configuration[ov::inference_num_threads.name()] =
        (m_cfg.mixed_lengths || m_cfg.invalid != SparsePAInvalid::kNone) ? 2 : 1;
    if (m_inference_precision == ov::element::bf16) {
        rel_threshold = 5e-2F;
        abs_threshold = 5e-2F;
    } else if (m_inference_precision == ov::element::f16) {
        rel_threshold = 5e-3F;
        abs_threshold = 5e-3F;
    }
    // ConvertPagedAttnInputs applies the KV cache precision (u8 by default) to the cache inputs: pin it to the data
    // precision for plain caches, or to the quantized precision, quantization mode and group size under test.
    if (quantized()) {
        configuration[ov::key_cache_precision.name()] = m_cfg.cache_precision;
        configuration[ov::value_cache_precision.name()] = m_cfg.cache_precision;
        configuration[ov::key_cache_group_size.name()] = static_cast<uint64_t>(m_cfg.group_size);
        configuration[ov::value_cache_group_size.name()] = static_cast<uint64_t>(m_cfg.group_size);
        configuration[ov::internal::key_cache_quant_mode.name()] =
            m_cfg.key_bychannel ? ov::internal::CacheQuantMode::BY_CHANNEL : ov::internal::CacheQuantMode::BY_TOKEN;
        configuration[ov::internal::value_cache_quant_mode.name()] = ov::internal::CacheQuantMode::BY_TOKEN;
    } else {
        // The cache inputs are declared at the data precision; ConvertPagedAttnInputs gives them the cache type.
        configuration.insert(ov::hint::kv_cache_precision(cache_type()));
    }

    const size_t Hkv = m_cfg.num_kv_heads;
    const size_t Dv = m_cfg.v_head_size;
    const size_t Hs = m_cfg.num_selection_heads;
    const size_t K = m_cfg.max_selected_blocks;
    const size_t n_phys = m_num_sequences * m_pages_per_sequence;
    const auto dyn = ov::Dimension::dynamic();
    const auto dim = [](size_t value) {
        return static_cast<int64_t>(value);
    };

    // The cache shape the model declares: the final one, or for an invalid case the dynamic page / head dims of a real
    // model (the invalid tensors differ in them).
    const auto declared_cache_shape = [&](bool is_key) {
        const auto shape = cache_shape(is_key);
        if (m_cfg.invalid == SparsePAInvalid::kNone || m_cfg.invalid == SparsePAInvalid::kBlockIdOutOfRange ||
            m_cfg.invalid == SparsePAInvalid::kNegativeBlockId ||
            m_cfg.invalid == SparsePAInvalid::kSubsequenceBeginsShort) {
            return ov::PartialShape(shape);
        }
        return ov::PartialShape{dim(shape[0]), dyn, dyn, dyn};
    };
    auto p_query = std::make_shared<ov::op::v0::Parameter>(data_precision,
                                                           ov::PartialShape{dyn, dim(m_num_heads * m_k_head_size)});
    auto p_key =
        std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{dyn, dim(Hkv * m_k_head_size)});
    auto p_value = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{dyn, dim(Hkv * Dv)});
    // Quantized caches get their final element type and padded shape from ConvertPagedAttnInputs (driven by the head
    // runtime info set on the op below), so they start out unspecified.
    auto p_key_cache = quantized()
                           ? std::make_shared<ov::op::v0::Parameter>(ov::element::dynamic, ov::PartialShape::dynamic(4))
                           : std::make_shared<ov::op::v0::Parameter>(data_precision, declared_cache_shape(true));
    auto p_value_cache =
        quantized() ? std::make_shared<ov::op::v0::Parameter>(ov::element::dynamic, ov::PartialShape::dynamic(4))
                    : std::make_shared<ov::op::v0::Parameter>(data_precision, declared_cache_shape(false));
    auto p_past_lens = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences});
    auto p_subsequence_begins =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences + 1});
    auto p_block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{n_phys});
    auto p_block_indices_begins =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences + 1});
    auto p_sel_indices =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{dyn, dim(Hs), dim(K)});
    auto p_sel_count = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{dyn, dim(Hs)});

    // Input order follows ov::op::internal::PagedSparseAttention; SCALE and SINKS are constants
    // (SINKS is an empty tensor: sinks are not supported yet).
    ov::ParameterVector parameters{p_query,
                                   p_key,
                                   p_value,
                                   p_key_cache,
                                   p_value_cache,
                                   p_past_lens,
                                   p_subsequence_begins,
                                   p_block_indices,
                                   p_block_indices_begins,
                                   p_sel_indices,
                                   p_sel_count};

    // Static target shapes of every step, in parameter order.
    std::vector<InputShape> input_shapes;
    for (const auto& parameter : parameters) {
        input_shapes.emplace_back(parameter->get_partial_shape(), std::vector<ov::Shape>{});
    }
    for (const auto& step : m_steps) {
        size_t tokens = 0;
        for (size_t s = 0; s < m_num_sequences; s++) {
            tokens += sequence_len(step[0], s);
        }
        const std::vector<ov::Shape> step_shapes = {{tokens, m_num_heads * m_k_head_size},
                                                    {tokens, Hkv * m_k_head_size},
                                                    {tokens, Hkv * Dv},
                                                    cache_shape(true),
                                                    cache_shape(false),
                                                    {m_num_sequences},
                                                    {m_num_sequences + 1},
                                                    {n_phys},
                                                    {m_num_sequences + 1},
                                                    {tokens, Hs, K},
                                                    {tokens, Hs}};
        for (size_t i = 0; i < step_shapes.size(); i++) {
            input_shapes[i].second.push_back(step_shapes[i]);
        }
    }
    init_input_shapes(input_shapes);

    auto scale = ov::op::v0::Constant::create(data_precision, ov::Shape{}, {m_scale});
    auto sinks = ov::op::v0::Constant::create(data_precision, ov::Shape{0}, std::vector<float>{});
    ov::OutputVector args = {p_query,
                             p_key,
                             p_value,
                             p_key_cache,
                             p_value_cache,
                             p_past_lens,
                             p_subsequence_begins,
                             p_block_indices,
                             p_block_indices_begins,
                             p_sel_indices,
                             p_sel_count,
                             scale,
                             sinks};
    auto sparse_pa = std::make_shared<ov::op::internal::PagedSparseAttention>(args, m_cfg.key_block_size);
    // Head metadata, as the transformation that creates the op records it: the node takes the head counts and sizes
    // from it, and ConvertPagedAttnInputs builds the cache shapes (32-token pages, padded when quantized) from it.
    sparse_pa->get_rt_info()["num_k_heads"] = Hkv;
    sparse_pa->get_rt_info()["k_head_size"] = m_k_head_size;
    sparse_pa->get_rt_info()["num_v_heads"] = Hkv;
    sparse_pa->get_rt_info()["v_head_size"] = Dv;

    // A Convert behind the output keeps the op out of the tail of the graph, see inference_precision.
    const ov::Output<ov::Node> output =
        m_inference_precision != data_precision
            ? std::make_shared<ov::op::v0::Convert>(sparse_pa, m_inference_precision)->output(0)
            : sparse_pa->output(0);
    function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(output)}, parameters);
}

void SparsePALayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    host_inputs.clear();

    const size_t step = m_step++;
    const size_t B = m_num_sequences;
    const size_t Hq = m_num_heads;
    const size_t Dh = m_k_head_size;
    const size_t Hkv = m_cfg.num_kv_heads;
    const size_t Dv = m_cfg.v_head_size;
    const size_t Hs = m_cfg.num_selection_heads;
    const size_t K = m_cfg.max_selected_blocks;
    const size_t Bk = m_cfg.key_block_size;
    const size_t n_phys = B * m_pages_per_sequence;
    // New tokens of every sequence in this step and the first global token of each: sequences of a mixed batch differ.
    std::vector<size_t> begins(B + 1, 0);
    m_cur_past = m_total_len;
    m_cur_len.resize(B);
    for (size_t s = 0; s < B; s++) {
        m_cur_len[s] = sequence_len(m_steps.at(step)[0], s);
        begins[s + 1] = begins[s] + m_cur_len[s];
        m_total_len[s] += m_cur_len[s];
    }
    const size_t tokens = begins[B];

    const auto make_i32 = [](const std::vector<int32_t>& values) {
        ov::Tensor t(ov::element::i32, ov::Shape{values.size()});
        std::copy(values.begin(), values.end(), t.data<int32_t>());
        return t;
    };

    // The caches start out empty and are written by the op itself in every step, like a real multi-call sequence.
    if (step == 0) {
        m_key_cache = ov::Tensor(cache_type(), cache_shape(true));
        m_value_cache = ov::Tensor(cache_type(), cache_shape(false));
        std::memset(m_key_cache.data(), 0, m_key_cache.get_byte_size());
        std::memset(m_value_cache.data(), 0, m_value_cache.get_byte_size());
    }

    // The values differ per step and along the tokens, so every sequence gets its own.
    const auto step_seed = static_cast<uint32_t>(step) * 101U;
    const auto make_data = [this](const ov::Shape& shape, uint32_t seed) {
        return ov::test::utils::create_and_fill_tensor(m_data_precision,
                                                       shape,
                                                       ov::test::utils::InputGenerateData(-1, 2, 10000, seed));
    };
    const auto query = make_data(ov::Shape{tokens, Hq * Dh}, 1 + step_seed);
    const auto key = make_data(ov::Shape{tokens, Hkv * Dh}, 2 + step_seed);
    const auto value = make_data(ov::Shape{tokens, Hkv * Dv}, 3 + step_seed);

    // Non-identity permutation (simple reversal) across all physical blocks shared by every sequence, so a kernel bug
    // that walked physical blocks in logical order instead of resolving through block_indices would give wrong results.
    std::vector<int32_t> block_indices(n_phys);
    for (size_t i = 0; i < n_phys; i++) {
        block_indices[i] = static_cast<int32_t>(n_phys - 1 - i);
    }

    // Synthetic selection of the complete blocks every token attends to, see make_selection().
    std::vector<int32_t> sel_indices_flat(tokens * Hs * K, -1);
    std::vector<int32_t> sel_count_flat(tokens * Hs, 0);
    std::mt19937 rng(12345U + static_cast<uint32_t>(step));
    for (size_t s = 0; s < B; s++) {
        for (size_t l = 0; l < m_cur_len[s]; l++) {
            const size_t token = begins[s] + l;
            const size_t num_complete = (m_cur_past[s] + l + 1) / Bk;
            for (size_t sh = 0; sh < Hs; sh++) {
                const size_t row = token * Hs + sh;
                make_selection(m_cfg.selection,
                               m_cfg.shuffle_blocks,
                               K,
                               num_complete,
                               rng,
                               &sel_indices_flat[row * K],
                               sel_count_flat[row]);
            }
        }
    }

    const auto& params = function->get_parameters();
    OPENVINO_ASSERT(targetInputStaticShapes.size() == params.size());
    std::vector<ov::Tensor> tensors(params.size());
    tensors[0] = query;
    tensors[1] = key;
    tensors[2] = value;
    tensors[3] = m_key_cache;
    tensors[4] = m_value_cache;
    auto cache_with_shape = [this](ov::Shape shape) {
        ov::Tensor t(cache_type(), shape);
        std::memset(t.data(), 0, t.get_byte_size());
        return t;
    };
    switch (m_cfg.invalid) {
    case SparsePAInvalid::kValueCachePageSize: {
        auto shape = cache_shape(false);
        shape[2] /= 2;
        tensors[4] = cache_with_shape(shape);
        break;
    }
    case SparsePAInvalid::kValueCacheHeadSize: {
        auto shape = cache_shape(false);
        shape[3] /= 2;
        tensors[4] = cache_with_shape(shape);
        break;
    }
    case SparsePAInvalid::kKvHeadCount: {
        auto key_shape = cache_shape(true);
        auto value_shape = cache_shape(false);
        key_shape[1]--;
        value_shape[1]--;
        tensors[3] = cache_with_shape(key_shape);
        tensors[4] = cache_with_shape(value_shape);
        break;
    }
    case SparsePAInvalid::kBlockIdOutOfRange:
        block_indices[0] = static_cast<int32_t>(n_phys);
        break;
    case SparsePAInvalid::kNegativeBlockId:
        block_indices[0] = -1;
        break;
    default:
        break;
    }
    tensors[5] = make_i32(std::vector<int32_t>(m_cur_past.begin(), m_cur_past.end()));
    std::vector<int32_t> subsequence_begins(B + 1);
    std::vector<int32_t> block_indices_begins(B + 1);
    for (size_t s = 0; s <= B; s++) {
        subsequence_begins[s] = static_cast<int32_t>(begins[s]);
        block_indices_begins[s] = static_cast<int32_t>(s * m_pages_per_sequence);
    }
    if (m_cfg.invalid == SparsePAInvalid::kSubsequenceBeginsShort) {
        subsequence_begins[B]--;
    }
    tensors[6] = make_i32(subsequence_begins);
    tensors[7] = make_i32(block_indices);
    tensors[8] = make_i32(block_indices_begins);
    {
        ov::Tensor t(ov::element::i32, ov::Shape{tokens, Hs, K});
        std::copy(sel_indices_flat.begin(), sel_indices_flat.end(), t.data<int32_t>());
        tensors[9] = t;
    }
    {
        ov::Tensor t(ov::element::i32, ov::Shape{tokens, Hs});
        std::copy(sel_count_flat.begin(), sel_count_flat.end(), t.data<int32_t>());
        tensors[10] = t;
    }

    for (size_t i = 0; i < params.size(); i++) {
        host_inputs[params[i]] = tensors[i];
        inputs[params[i]] = tensors[i];
    }
}

std::vector<ov::Tensor> SparsePALayerCPUTest::calculate_refs() {
    const auto& params = function->get_parameters();
    const auto to_f32 = [&](const std::shared_ptr<ov::Node>& p) {
        return read_tensor_to_f32(host_inputs.at(p));
    };
    const auto to_i32 = [&](const std::shared_ptr<ov::Node>& p) {
        const auto& t = host_inputs.at(p);
        const auto* ptr = t.data<const int32_t>();
        return std::vector<int32_t>(ptr, ptr + t.get_size());
    };

    const size_t B = m_num_sequences;
    std::vector<size_t> begins(B + 1, 0);
    for (size_t s = 0; s < B; s++) {
        begins[s + 1] = begins[s] + m_cur_len[s];
    }
    const size_t Hq = m_num_heads;
    const size_t Dh = m_k_head_size;
    const size_t Hkv = m_cfg.num_kv_heads;
    const size_t Dv = m_cfg.v_head_size;
    const size_t Hs = m_cfg.num_selection_heads;
    const size_t K = m_cfg.max_selected_blocks;
    const size_t page = m_cfg.page_size;
    const auto prec = m_cfg.cache_precision;
    const ReferenceDims dims{Hq, Hkv, Hs, Dh, Dv, m_cfg.key_block_size, K};

    const auto query = to_f32(params[0]);
    const auto key = to_f32(params[1]);
    const auto value = to_f32(params[2]);
    const auto sel_indices = to_i32(params[9]);
    const auto sel_count = to_i32(params[10]);

    if (m_eff_key.empty()) {
        m_eff_key.assign(B, std::vector<std::vector<float>>(Hkv));
        m_eff_value.assign(B, std::vector<std::vector<float>>(Hkv));
    }

    for (size_t s = 0; s < B; s++) {
        const size_t L = m_cur_len[s];
        const size_t past = m_cur_past[s];
        const size_t total = past + L;
        for (size_t kv = 0; kv < Hkv; kv++) {
            auto& eff_key = m_eff_key[s][kv];
            auto& eff_value = m_eff_value[s][kv];
            eff_key.resize(total * Dh);
            eff_value.resize(total * Dv);
            for (size_t l = 0; l < L; l++) {
                const size_t token = begins[s] + l;
                const float* k_src = &key[(token * Hkv + kv) * Dh];
                const float* v_src = &value[(token * Hkv + kv) * Dv];
                float* k_dst = &eff_key[(past + l) * Dh];
                float* v_dst = &eff_value[(past + l) * Dv];
                if (quantized()) {
                    fake_quant_by_token(prec, m_cfg.group_size, v_src, v_dst, Dv);
                    if (m_cfg.key_bychannel) {
                        std::copy_n(k_src, Dh, k_dst);  // requantized with its page below
                    } else {
                        fake_quant_by_token(prec, m_cfg.group_size, k_src, k_dst, Dh);
                    }
                } else {
                    for (size_t d = 0; d < Dh; d++) {
                        k_dst[d] = through_cache(k_src[d]);
                    }
                    for (size_t d = 0; d < Dv; d++) {
                        v_dst[d] = through_cache(v_src[d]);
                    }
                }
            }
            if (quantized() && m_cfg.key_bychannel) {
                for (size_t page_start = (past / page) * page; page_start < total; page_start += page) {
                    const size_t rows = std::min(page, total - page_start);
                    const std::vector<float> page_rows(
                        eff_key.begin() + static_cast<ptrdiff_t>(page_start * Dh),
                        eff_key.begin() + static_cast<ptrdiff_t>((page_start + rows) * Dh));
                    fake_quant_by_channel(prec, page_rows.data(), &eff_key[page_start * Dh], rows, Dh);
                }
            }
        }
    }

    ov::Tensor output(ov::element::f32, ov::Shape{begins[B], Hq * Dv});
    auto* output_data = output.data<float>();
    for (size_t s = 0; s < B; s++) {
        const size_t L = m_cur_len[s];
        const size_t past = m_cur_past[s];
        // Full logical history of the sequence as [kv, total, S].
        std::vector<float> history_key;
        std::vector<float> history_value;
        for (size_t kv = 0; kv < Hkv; kv++) {
            history_key.insert(history_key.end(), m_eff_key[s][kv].begin(), m_eff_key[s][kv].end());
            history_value.insert(history_value.end(), m_eff_value[s][kv].begin(), m_eff_value[s][kv].end());
        }
        const auto first = static_cast<ptrdiff_t>(begins[s]);
        const auto last = first + static_cast<ptrdiff_t>(L);
        const std::vector<float> query_s(query.begin() + first * static_cast<ptrdiff_t>(Hq * Dh),
                                         query.begin() + last * static_cast<ptrdiff_t>(Hq * Dh));
        const std::vector<int32_t> sel_indices_s(sel_indices.begin() + first * static_cast<ptrdiff_t>(Hs * K),
                                                 sel_indices.begin() + last * static_cast<ptrdiff_t>(Hs * K));
        const std::vector<int32_t> sel_count_s(sel_count.begin() + first * static_cast<ptrdiff_t>(Hs),
                                               sel_count.begin() + last * static_cast<ptrdiff_t>(Hs));
        const auto out_s = run_sparse_pa_reference(dims,
                                                   L,
                                                   m_scale,
                                                   static_cast<int64_t>(past),
                                                   query_s,
                                                   history_key,
                                                   history_value,
                                                   sel_indices_s,
                                                   sel_count_s);
        std::copy(out_s.begin(), out_s.end(), output_data + begins[s] * Hq * Dv);
    }
    return {output};
}

void SparsePALayerCPUTest::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
}

// The node runs in the inference precision whatever the data precision is.
void SparsePALayerCPUTest::check_runtime_precision() const {
    const auto runtime_model = compiledModel.get_runtime_model();
    bool found = false;
    for (const auto& op : runtime_model->get_ops()) {
        const auto& info = op->get_rt_info();
        const auto layer_type = info.find(ov::exec_model_info::LAYER_TYPE);
        if (layer_type != info.end() && layer_type->second.as<std::string>() == "SparsePA") {
            found = true;
            EXPECT_EQ(info.at(ov::exec_model_info::RUNTIME_PRECISION).as<std::string>(),
                      m_inference_precision.get_type_name());
        }
    }
    EXPECT_TRUE(found) << "no SparsePA node in the runtime model";
}

TEST_P(SparsePALayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto invalid = std::get<2>(GetParam()).invalid;
    if (invalid != SparsePAInvalid::kNone) {
        // Inference must reject the input with an exception, before anything is written or read out of bounds.
        compile_model();
        generate_inputs(targetStaticShapes.at(0));
        try {
            infer();
            ADD_FAILURE() << "the invalid input was accepted";
        } catch (const std::exception& ex) {
            // A tensor of a wrong cache shape is already refused when it is set (the plugin pins the cache shapes of
            // the model), the page table by the node.
            const std::string what = ex.what();
            const std::string expected =
                invalid == SparsePAInvalid::kSubsequenceBeginsShort ? "must cover the tokens"
                : invalid == SparsePAInvalid::kBlockIdOutOfRange || invalid == SparsePAInvalid::kNegativeBlockId
                    ? "block id"
                    : "incompatible";
            EXPECT_NE(what.find(expected), std::string::npos) << what;
        }
        return;
    }
    run();
    check_runtime_precision();
}

}  // namespace test
}  // namespace ov
