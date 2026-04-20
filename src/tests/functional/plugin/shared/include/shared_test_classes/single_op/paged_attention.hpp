// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <map>
#include <set>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

#include "openvino/op/paged_attention.hpp"


namespace ov {
namespace test {

using InputShapes = std::vector<InputShape>;
using PagedAttentionTestParams = std::tuple<ElementType, InputShapes, bool, bool, bool, int32_t, bool, int32_t, ov::AnyMap>;

class PagedAttentionLayerTest
    : public testing::WithParamInterface<PagedAttentionTestParams>,
      public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttentionTestParams>& obj) {
        const auto& [inType,
                     inputShapes,
                     extendBlockIndices,
                     enableXattn,
                     sinkInput,
                     slidingWindow,
                     useAlibi,
                     maxContextLen,
                     additional_config] = obj.param;
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
        result << "ExtendBlockIndices=" << extendBlockIndices << "_";
        result << "EnableXattn=" << enableXattn << "_";
        result << "SinkInput=" << sinkInput << "_";
        result << "SlidingWindow=" << slidingWindow << "_";
        result << "UseAlibi=" << useAlibi << "_";
        result << "MaxContextLen=" << maxContextLen << "_";
        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second.as<std::string>() << "_";
        }
        result << ")";

        return result.str();
    }

protected:
    // Model construction
    static std::shared_ptr<ov::op::v0::Parameter> make_param(const ov::PartialShape& ps,
                                                             ov::element::Type et,
                                                             const std::string& name) {
        auto p = std::make_shared<ov::op::v0::Parameter>(et, ps);
        p->set_friendly_name(name);
        p->get_output_tensor(0).set_names({name});
        return p;
    }

    std::shared_ptr<ov::Model> make_paged_attn_model(ov::element::Type data_type,
                                                     bool enable_xattn,
                                                     int64_t head_size = 64,
                                                     int64_t head_num = 8,
                                                     bool use_sink_input = false,
                                                     int32_t sliding_window = 0,
                                                     bool use_alibi = false,
                                                     int32_t max_ctx_len = 1024,
                                                     bool use_rotation = false,
                                                     int64_t block_size = 32,
                                                     int32_t adaptive_rkv_eviction_size = 0,
                                                     int32_t score_window_val = -1) {
        // PA expects q/k/v as [tokens, features]
        auto q = make_param({ov::Dimension::dynamic(), ov::Dimension::dynamic()}, data_type, "q");
        auto k = make_param({ov::Dimension::dynamic(), head_num * head_size}, data_type, "k");
        auto v = make_param({ov::Dimension::dynamic(), head_num * head_size}, data_type, "v");

        // Cache layout: [num_blocks, num_kv_heads, block_size, head_size]
        // Use data_type directly so TEMPLATE can allocate real tensors without running ConvertPagedAttnInputs
        auto key_cache   = make_param({ov::Dimension::dynamic(), head_num, block_size, head_size}, data_type, "key_cache.0");
        auto value_cache = make_param({ov::Dimension::dynamic(), head_num, block_size, head_size}, data_type, "value_cache.0");

        auto past_lens = make_param({ov::Dimension::dynamic()}, ov::element::i32, "past_lens");
        auto subseq_begins = make_param({ov::Dimension::dynamic()}, ov::element::i32, "subsequence_begins");
        auto block_indices = make_param({ov::Dimension::dynamic()}, ov::element::i32, "block_indices");
        auto block_indices_begins = make_param({ov::Dimension::dynamic()}, ov::element::i32, "block_indices_begins");

        // Use typed empty vectors for zero-element constants to avoid
        // ambiguity with Constant::create overload resolution
        const float scale_value = 1.0f / std::sqrt(static_cast<float>(head_size));
        auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale_value});
        auto sliding_windows = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{sliding_window});
        std::shared_ptr<ov::op::v0::Constant> alibi_slopes;
        if (use_alibi) {
            // Generate slopes: 1.0 / 2^i  for i = 0..head_num-1
            std::vector<float> slopes(static_cast<size_t>(head_num));
            for (int64_t i = 0; i < head_num; ++i) {
                slopes[i] = 1.0f / static_cast<float>(1 << i);
            }
            alibi_slopes = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                ov::Shape{static_cast<size_t>(head_num)}, slopes);
        } else {
            alibi_slopes = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0}, std::vector<float>{});
        }
        auto max_context_len = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{max_ctx_len});
        // score_window_val < 0 -> use max_ctx_len (all tokens, positive window); 0 -> disabled
        const int32_t effective_score_window = (score_window_val < 0) ? max_ctx_len : score_window_val;
        auto score_aggregation_window = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{effective_score_window});

        std::shared_ptr<ov::op::v0::Constant> rotated_block_indices;
        std::shared_ptr<ov::op::v0::Constant> rotation_deltas;
        std::shared_ptr<ov::op::v0::Constant> rotation_trig_lut;
        if (use_rotation) {
            // Rotate block 0 with a simple RoPE-style trig LUT
            // We use per-block granularity: rotation_deltas shape [1, 1] (1 block, delta = 0 => LUT row 0)
            rotated_block_indices = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});

            // Per-block delta: single block, single delta pointing to LUT row 0
            rotated_block_indices_ = std::vector<int32_t>{0};
            rotation_deltas = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1}, std::vector<int32_t>{0});

            // Trig LUT: [1, head_size] with layout [cos_0..cos_{half-1}, sin_0..sin_{half-1}]
            // Use a gentle rotation: cos ~= 0.9, sin ~= 0.1 (not unit-circle exact, but valid for testing)
            const size_t hs = static_cast<size_t>(head_size);
            const size_t half = hs / 2;
            std::vector<float> lut(hs, 0.f);
            for (size_t d = 0; d < half; ++d) {
                // Use RoPE-like frequencies: theta_d = 1 / 10000^(2d/head_size)
                // Apply a rotation of delta_position = 1
                const float theta = 1.f / std::pow(10000.f, 2.f * static_cast<float>(d) / static_cast<float>(hs));
                lut[d] = std::cos(theta);          // cos part
                lut[half + d] = std::sin(theta);   // sin part
            }
            rotation_trig_lut = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, hs}, lut);
        } else {
            rotated_block_indices = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{0});
            rotation_deltas       = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{0});
            rotation_trig_lut     = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0}, std::vector<float>{0});
        }

        std::shared_ptr<ov::op::v0::Constant> xattention_threshold;
        if (enable_xattn) {
            xattention_threshold = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0.9f});
        } else {
            xattention_threshold = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0}, std::vector<float>{0});
        }
        auto xattention_block_size = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{64});
        auto xattention_stride     = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{8});

        // Sink input: [1, H, 1, 1] when enabled, empty [0] when disabled
        // Matching the pattern used by the CPU-specific test
        std::shared_ptr<ov::op::v0::Constant> sinks;
        if (use_sink_input) {
            // Use per-head sink values comparable in magnitude to attention logits
            // so the effect is clearly visible: 3.0 + 0.5 * h
            const size_t hn = static_cast<size_t>(head_num);
            std::vector<float> sink_data(hn);
            for (size_t h = 0; h < hn; ++h) {
                sink_data[h] = 3.0f + 0.5f * static_cast<float>(h);
            }
            sinks = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, hn, 1, 1}, sink_data);
        } else {
            sinks = std::make_shared<ov::op::v0::Constant>(data_type, ov::Shape{0}, std::vector<float>{});
        }

        // adaptive_rkv inputs
        std::shared_ptr<ov::op::v0::Constant> adaptive_rkv_start_size;
        std::shared_ptr<ov::op::v0::Constant> adaptive_rkv_evictable_sizes;
        std::shared_ptr<ov::op::v0::Constant> adaptive_rkv_diversity_block_set_indices;
        std::shared_ptr<ov::op::v0::Constant> adaptive_rkv_diversity_block_set_indices_begins;
        if (adaptive_rkv_eviction_size > 0) {
            // start_size = 0 for simplicity
            adaptive_rkv_start_size = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{}, std::vector<int32_t>{0});
            // Single sequence: evictable_sizes = [eviction_size]
            adaptive_rkv_evictable_sizes = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{1}, std::vector<int32_t>{adaptive_rkv_eviction_size});
            // block_set_indices: list of block indices in the eviction zone (0..num_eviction_blocks-1)
            const int32_t num_eviction_blocks = adaptive_rkv_eviction_size / static_cast<int32_t>(block_size);
            std::vector<int32_t> block_set(static_cast<size_t>(num_eviction_blocks));
            for (int32_t i = 0; i < num_eviction_blocks; ++i) block_set[i] = i;
            adaptive_rkv_diversity_block_set_indices = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{static_cast<size_t>(num_eviction_blocks)}, block_set);
            // begins: [0, num_eviction_blocks] for single sequence
            adaptive_rkv_diversity_block_set_indices_begins = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{2}, std::vector<int32_t>{0, num_eviction_blocks});
        } else {
            adaptive_rkv_start_size = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{}, std::vector<int32_t>{0});
            adaptive_rkv_evictable_sizes = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{0}, std::vector<int32_t>{0});
            adaptive_rkv_diversity_block_set_indices = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{0}, std::vector<int32_t>{0});
            adaptive_rkv_diversity_block_set_indices_begins = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{0}, std::vector<int32_t>{0});
        }

        ov::ParameterVector params = {q,k,v,key_cache,value_cache,past_lens,subseq_begins,block_indices,block_indices_begins};
        auto token_type_ids = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});
        auto qq_bias = std::make_shared<ov::op::v0::Constant>(
                ov::element::u8, ov::Shape{0}, std::vector<uint8_t>{});
        auto qq_bias_begins = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});
        ov::OutputVector pa_inputs = {q,k,v,key_cache,value_cache,past_lens,subseq_begins,block_indices,block_indices_begins,
                                      scale,sliding_windows,alibi_slopes,max_context_len,score_aggregation_window,
                                      rotated_block_indices,rotation_deltas,rotation_trig_lut,
                                      xattention_threshold,xattention_block_size,xattention_stride,
                                      sinks,
                                      adaptive_rkv_start_size,adaptive_rkv_evictable_sizes,
                                      adaptive_rkv_diversity_block_set_indices,adaptive_rkv_diversity_block_set_indices_begins,
                                      token_type_ids,
                                      qq_bias,qq_bias_begins};

        auto pa = std::make_shared<ov::op::PagedAttentionExtension>(pa_inputs);

        // Provide head metadata required by ConvertPagedAttnInputs transformation
        // These keys are the canonical ones read by the transformation callback
        auto& rt = pa->get_rt_info();
        rt["num_k_heads"] = static_cast<size_t>(head_num);
        rt["k_head_size"] = static_cast<size_t>(head_size);
        rt["num_v_heads"] = static_cast<size_t>(head_num);
        rt["v_head_size"] = static_cast<size_t>(head_size);

        // Output 2 (diversity scores) is added only when adaptive RKV is active, because
        // the CPU does not compute it and the buffer contents are unspecified on that path.
        ov::OutputVector model_outputs = {pa->output(0), pa->output(1)};
        if (adaptive_rkv_eviction_size > 0) {
            model_outputs.push_back(pa->output(2));
        }
        auto model = std::make_shared<ov::Model>(model_outputs, params, "pa_vsref");
        return model;
    }

    // Input generation like in the CPU plugin
    template <typename IT, typename T>
    static void strided_iota(IT first, size_t n, T value, T stride) {
        for (size_t i = 0; i < n; i++) {
            const float idx = static_cast<float>(n - 1 - i);
            *first++ = static_cast<T>(value + stride * static_cast<T>(idx));
        }
    }

    struct StepInputs {
        std::map<std::shared_ptr<ov::op::v0::Parameter>, ov::Tensor> tensors;
    };

    StepInputs make_step_inputs(const std::shared_ptr<ov::Model>& model,
                                const ov::Shape& lbhs,
                                int step_idx,
                                bool extendBlockIndices,
                                ov::Tensor& key_cache,
                                ov::Tensor& value_cache,
                                int32_t& past_len_count) {
        StepInputs out;
        auto params = model->get_parameters();

        const size_t L = lbhs[0], B = lbhs[1], H = lbhs[2], S = lbhs[3];
        const size_t tokens = L * B;
        const size_t feats = H * S;

        auto fill_fp = [&](ov::Tensor& t, float base) {
            if (t.get_element_type() == ov::element::f32)
                strided_iota(static_cast<float*>(t.data()), t.get_size(), base, 0.1f);
            else if (t.get_element_type() == ov::element::f16)
                strided_iota(static_cast<ov::float16*>(t.data()), t.get_size(), static_cast<ov::float16>(base), ov::float16(0.1f));
            else
                strided_iota(static_cast<ov::bfloat16*>(t.data()), t.get_size(), static_cast<ov::bfloat16>(base), ov::bfloat16(0.1f));
        };

        // q,k,v
        {
            ov::Tensor tq(params[0]->get_element_type(), {tokens, feats});
            ov::Tensor tk(params[1]->get_element_type(), {tokens, feats});
            ov::Tensor tv(params[2]->get_element_type(), {tokens, feats});
            fill_fp(tq, step_idx + 1.f);
            fill_fp(tk, step_idx + 2.f);
            fill_fp(tv, step_idx + 3.f);
            out.tensors[params[0]] = tq;
            out.tensors[params[1]] = tk;
            out.tensors[params[2]] = tv;
        }

        // cache tensors (already allocated outside)
        out.tensors[params[3]] = key_cache;
        out.tensors[params[4]] = value_cache;


        // past_lens/subseq/block tables
        const size_t block_size = key_cache.get_shape().at(2);
        const size_t batch_seq = B;

        // number of blocks needed PER SEQUENCE (batch item)
        const int32_t used_blocks_per_seq = std::max<int32_t>(
            1,
            static_cast<int32_t>((static_cast<int64_t>(past_len_count) + static_cast<int64_t>(L) +
                                  static_cast<int64_t>(block_size) - 1) /
                                 static_cast<int64_t>(block_size)));

        OPENVINO_ASSERT(static_cast<size_t>(used_blocks_per_seq) <= max_blocks_per_seq_,
                        "PagedAttention test: cache plan too small for current step");

        const int32_t bi_count_per_seq =
            extendBlockIndices ? std::max<int32_t>(2, used_blocks_per_seq) : used_blocks_per_seq;

        ov::Tensor past_lens(ov::element::i32, {batch_seq});
        ov::Tensor subseq(ov::element::i32, {batch_seq + 1});
        ov::Tensor bi_begins(ov::element::i32, {batch_seq + 1});
        ov::Tensor bi(ov::element::i32, {batch_seq * static_cast<size_t>(bi_count_per_seq)});

        auto* pl = past_lens.data<int32_t>();
        auto* sb = subseq.data<int32_t>();
        auto* bb = bi_begins.data<int32_t>();
        auto* b  = bi.data<int32_t>();

        for (size_t s = 0; s < batch_seq; ++s) {
            pl[s] = (step_idx == 0) ? 0 : past_len_count;
            sb[s] = static_cast<int32_t>(s * L);
            bb[s] = static_cast<int32_t>(s * static_cast<size_t>(bi_count_per_seq));

            const int32_t block_base = static_cast<int32_t>(s * max_blocks_per_seq_);
            for (int32_t i = 0; i < bi_count_per_seq; ++i) {
                b[bb[s] + i] = (i < used_blocks_per_seq) ? (block_base + i) : -1;
            }
        }

        sb[batch_seq] = static_cast<int32_t>(tokens);
        bb[batch_seq] = static_cast<int32_t>(batch_seq * static_cast<size_t>(bi_count_per_seq));

        out.tensors[params[5]] = past_lens;
        out.tensors[params[6]] = subseq;
        out.tensors[params[7]] = bi;
        out.tensors[params[8]] = bi_begins;

        past_len_count += static_cast<int32_t>(L);

        return out;
    }

public:
    std::vector<std::vector<ov::Tensor>> run_device(ov::Core& core,
                                                    const std::shared_ptr<ov::Model>& model,
                                                    const std::string& device,
                                                    ov::AnyMap cfg,
                                                    bool extendBlockIndices,
                                                    const std::vector<StepInputs>& steps) {
        auto compiled = core.compile_model(model, device, cfg);
        auto req = compiled.create_infer_request();
        const size_t num_outputs = compiled.outputs().size();

        std::vector<std::vector<ov::Tensor>> outs;
        outs.reserve(steps.size());

        for (size_t si = 0; si < steps.size(); ++si) {
            const auto& step = steps[si];
            for (const auto& kv : step.tensors) {
                req.set_tensor(kv.first, kv.second);
            }

            req.infer();

            std::vector<ov::Tensor> step_outs;
            step_outs.reserve(num_outputs);
            for (size_t oi = 0; oi < num_outputs; ++oi) {
                auto t = req.get_output_tensor(oi);
                ov::Tensor copy(t.get_element_type(), t.get_shape());
                t.copy_to(copy);
                step_outs.push_back(std::move(copy));
            }
            outs.push_back(std::move(step_outs));
        }
        return outs;
    }

    void SetUp() override {
        is_report_stages = true;
        const auto& [inType, inputShapes, extendBlockIndices, enableXattn, sinkInput, slidingWindow, useAlibi, maxContextLen, additional_config] = GetParam();

        init_input_shapes(inputShapes);

        // Derive H (num_heads) and S (head_size) from the first target shape [L,B,H,S]
        OPENVINO_ASSERT(!targetStaticShapes.empty() && targetStaticShapes[0][0].size() == 4,
                        "PagedAttention test expects [L,B,H,S] shapes");
        const int64_t head_num  = static_cast<int64_t>(targetStaticShapes[0][0][2]);
        const int64_t head_size = static_cast<int64_t>(targetStaticShapes[0][0][3]);

        // Check if rotation is requested via the additional config map
        use_rotation_ = false;
        {
            auto it = additional_config.find("test_use_rotation");
            if (it != additional_config.end()) {
                use_rotation_ = it->second.as<bool>();
            }
        }

        // Check for adaptive RKV parameters
        int64_t cfg_block_size = 32;
        int32_t cfg_arkv_eviction_size = 0;
        {
            auto it = additional_config.find("test_block_size");
            if (it != additional_config.end()) {
                cfg_block_size = static_cast<int64_t>(it->second.as<int>());
            }
        }
        {
            auto it = additional_config.find("test_adaptive_rkv_eviction_size");
            if (it != additional_config.end()) {
                cfg_arkv_eviction_size = static_cast<int32_t>(it->second.as<int>());
            }
        }

        pa_model_ = make_paged_attn_model(inType, enableXattn, head_size, head_num, sinkInput, slidingWindow, useAlibi, maxContextLen, use_rotation_, cfg_block_size, cfg_arkv_eviction_size);

        // Pre-generate the step inputs once, so the CPU/TEMPLATE see identical tensors
        // Allocate caches once here, and reuse for both runs by copying initial cache state
        {
            const auto& kc_ps = pa_model_->get_parameters()[3]->get_partial_shape();
            OPENVINO_ASSERT(kc_ps.rank().is_static() && kc_ps.rank().get_length() == 4,
                            "PagedAttention test: expected key_cache rank 4");
            OPENVINO_ASSERT(kc_ps[2].is_static(), "PagedAttention test: expected static block_size dimension in cache");
            block_size_ = static_cast<size_t>(kc_ps[2].get_length());

            // We treat each batch item as an independent sequence in metadata tensors
            // Ensure cache has enough blocks per sequence for the entire multi-step run
            batch_size_ = targetStaticShapes.empty() ? 1 : targetStaticShapes[0][0][1];
            max_blocks_per_seq_ = 1;
            int32_t past_tmp = 0;

            for (size_t i = 0; i < targetStaticShapes.size(); ++i) {
                const auto& lbhs = targetStaticShapes[i][0];
                OPENVINO_ASSERT(lbhs.size() == 4, "PagedAttention test expects [L,B,H,S] shapes");
                const size_t L = lbhs[0];
                const size_t B = lbhs[1];
                if (i == 0) {
                    batch_size_ = B;
                } else {
                    OPENVINO_ASSERT(B == batch_size_, "PagedAttention test expects stable batch across steps");
                }

                const size_t blocks_now =
                    std::max<size_t>(1, (static_cast<size_t>(past_tmp) + L + block_size_ - 1) / block_size_);
                max_blocks_per_seq_ = std::max(max_blocks_per_seq_, blocks_now);
                past_tmp += static_cast<int32_t>(L);
            }

            allocate_caches_for_model(pa_model_, batch_size_ * max_blocks_per_seq_, inType);
        }

        int32_t past = 0;
        steps_.clear();
        steps_.reserve(targetStaticShapes.size());
        for (size_t i = 0; i < targetStaticShapes.size(); ++i) {
            steps_.push_back(make_step_inputs(pa_model_, targetStaticShapes[i][0], static_cast<int>(i),
                                              extendBlockIndices, key_cache_init_, value_cache_init_, past));
        }
    }

    void allocate_caches_for_model(const std::shared_ptr<ov::Model>& model,
                                   size_t block_nums,
                                   ov::element::Type inType) {
        // Use model parameter shapes directly (not compiled model) for determinism
        auto params = model->get_parameters();
        auto kc = params[3];
        auto vc = params[4];

        auto ps_k = kc->get_partial_shape();
        auto ps_v = vc->get_partial_shape();
        OPENVINO_ASSERT(ps_k.rank().is_static() && ps_k.rank().get_length() == 4);
        OPENVINO_ASSERT(ps_v.rank().is_static() && ps_v.rank().get_length() == 4);

        ps_k[0] = static_cast<int64_t>(block_nums);
        ps_v[0] = static_cast<int64_t>(block_nums);

        // Allocate caches using the same element type as PA expects
        key_cache_init_ = ov::Tensor(inType, ps_k.get_shape());
        value_cache_init_ = ov::Tensor(inType, ps_v.get_shape());
        std::memset(key_cache_init_.data(), 0, key_cache_init_.get_byte_size());
        std::memset(value_cache_init_.data(), 0, value_cache_init_.get_byte_size());
    }

protected:
    size_t block_size_ = 32;
    size_t max_blocks_per_seq_ = 1;
    size_t batch_size_ = 1;
    bool use_rotation_ = false;

    std::shared_ptr<ov::Model> pa_model_;
    ov::Tensor key_cache_init_;
    ov::Tensor value_cache_init_;
    std::vector<StepInputs> steps_;
    std::vector<int32_t> rotated_block_indices_;  // stored for debug prints
};

}  // namespace test
}  // namespace ov
