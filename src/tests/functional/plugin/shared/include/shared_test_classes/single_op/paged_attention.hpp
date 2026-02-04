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
using PagedAttentionTestParams = std::tuple<ElementType, InputShapes, bool, bool, bool, int32_t, ov::AnyMap>;

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
        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second.as<std::string>() << "_";
        }
        result << ")";

        return result.str();
    }

private:
    // ---------- Model construction (PA only) ----------
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
                                                     int32_t sliding_window = 0) {
        // PA expects q/k/v as [tokens, features]
        auto q = make_param({ov::Dimension::dynamic(), head_num * head_size}, data_type, "q");
        auto k = make_param({ov::Dimension::dynamic(), head_num * head_size}, data_type, "k");
        auto v = make_param({ov::Dimension::dynamic(), head_num * head_size}, data_type, "v");

        // IMPORTANT: cache should match PA expectation (use rank-4 here)
        // [num_blocks, num_kv_heads, block_size, head_size]
        auto key_cache   = make_param({ov::Dimension::dynamic(), head_num, 32, head_size}, ov::element::dynamic, "key_cache.0");
        auto value_cache = make_param({ov::Dimension::dynamic(), head_num, 32, head_size}, ov::element::dynamic, "value_cache.0");

        auto past_lens = make_param({ov::Dimension::dynamic()}, ov::element::i32, "past_lens");
        auto subseq_begins = make_param({ov::Dimension::dynamic()}, ov::element::i32, "subsequence_begins");
        auto block_indices = make_param({ov::Dimension::dynamic()}, ov::element::i32, "block_indices");
        auto block_indices_begins = make_param({ov::Dimension::dynamic()}, ov::element::i32, "block_indices_begins");

        const float scale_value = 1.0f / std::sqrt(static_cast<float>(head_size));
        auto scale = ov::op::v0::Constant::create(ov::element::f32, {}, {scale_value});
        auto sliding_windows = ov::op::v0::Constant::create(ov::element::i32, {}, {sliding_window});
        auto alibi_slopes = ov::op::v0::Constant::create(ov::element::f32, {0}, {});
        auto max_context_len = ov::op::v0::Constant::create(ov::element::i32, {}, {1024});
        auto score_aggregation_window = ov::op::v0::Constant::create(ov::element::i32, {}, {0});

        auto rotated_block_indices = ov::op::v0::Constant::create(ov::element::i32, {0}, {});
        auto rotation_deltas      = ov::op::v0::Constant::create(ov::element::i32, {0}, {});
        auto rotation_trig_lut    = ov::op::v0::Constant::create(ov::element::f32, {0}, {});

        auto xattention_threshold = enable_xattn
            ? ov::op::v0::Constant::create(ov::element::f32, {1}, {0.9f})
            : ov::op::v0::Constant::create(ov::element::f32, {0}, {});
        auto xattention_block_size = ov::op::v0::Constant::create(ov::element::i32, {}, {64});
        auto xattention_stride     = ov::op::v0::Constant::create(ov::element::i32, {}, {8});

        auto sinks = ov::op::v0::Constant::create(data_type, {0}, {});

        // adaptive_rkv inputs (ignored)
        auto adaptive_rkv_start_size = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
        auto adaptive_rkv_evictable_sizes = ov::op::v0::Constant::create(ov::element::i32, {0}, {});
        auto adaptive_rkv_diversity_block_set_indices = ov::op::v0::Constant::create(ov::element::i32, {0}, {});
        auto adaptive_rkv_diversity_block_set_indices_begins = ov::op::v0::Constant::create(ov::element::i32, {0}, {});

        ov::ParameterVector params = {q,k,v,key_cache,value_cache,past_lens,subseq_begins,block_indices,block_indices_begins};
        ov::OutputVector pa_inputs = {q,k,v,key_cache,value_cache,past_lens,subseq_begins,block_indices,block_indices_begins,
                                      scale,sliding_windows,alibi_slopes,max_context_len,score_aggregation_window,
                                      rotated_block_indices,rotation_deltas,rotation_trig_lut,
                                      xattention_threshold,xattention_block_size,xattention_stride,
                                      sinks,
                                      adaptive_rkv_start_size,adaptive_rkv_evictable_sizes,
                                      adaptive_rkv_diversity_block_set_indices,adaptive_rkv_diversity_block_set_indices_begins};

        auto pa = std::make_shared<ov::op::PagedAttentionExtension>(pa_inputs);
        return std::make_shared<ov::Model>(ov::OutputVector{pa->output(0)}, params, "pa_vsref");
    }

    // ---------- Input generation ----------
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
        const size_t batch_seq = 1;
        const int32_t total_blocks = static_cast<int32_t>((tokens + past_len_count + 32 - 1) / 32); // divide UP

        ov::Tensor past_lens(ov::element::i32, {batch_seq});
        ov::Tensor subseq(ov::element::i32, {batch_seq + 1});
        ov::Tensor bi_begins(ov::element::i32, {batch_seq + 1});
        ov::Tensor bi(ov::element::i32, {static_cast<size_t>(std::max<int32_t>(1, total_blocks))});

        auto* pl = past_lens.data<int32_t>();
        auto* sb = subseq.data<int32_t>();
        auto* bb = bi_begins.data<int32_t>();
        auto* b  = bi.data<int32_t>();

        if (step_idx == 0) {
            pl[0] = 0;
            sb[0] = 0;
            sb[1] = static_cast<int32_t>(L);
            bb[0] = 0;
            bb[1] = extendBlockIndices ? 2 : total_blocks; // vLLM-style over-allocation
            for (int32_t i = 0; i < total_blocks; i++) b[i] = i;
        } else {
            pl[0] = past_len_count;
            sb[0] = 0;
            sb[1] = static_cast<int32_t>(L);
            bb[0] = 0;
            bb[1] = total_blocks;
            for (int32_t i = 0; i < total_blocks; i++) b[i] = i;
        }

        out.tensors[params[5]] = past_lens;
        out.tensors[params[6]] = subseq;
        out.tensors[params[7]] = bi;
        out.tensors[params[8]] = bi_begins;

        past_len_count += static_cast<int32_t>(L);
        return out;
    }

public:
    // ---------- Execution ----------
    std::vector<ov::Tensor> run_device(const std::shared_ptr<ov::Model>& model,
                                       const std::string& device,
                                       ov::AnyMap cfg,
                                       bool extendBlockIndices,
                                       const std::vector<StepInputs>& steps) {
        ov::Core core;
        //// =====================

        for (size_t i = 0; i < model->inputs().size(); ++i) {
            const auto& p = model->input(i);
            std::cerr << "MODEL IN[" << i << "] " << p.get_any_name()
                    << " " << p.get_element_type()
                    << " " << p.get_partial_shape() << "\n";
        }
        std::cerr << "CPU supported_properties:\n";
        auto props = core.get_property("CPU", ov::supported_properties);

for (const ov::PropertyName& p : props) {
    std::cerr << p << std::endl;
}
        //// =====================

        auto compiled = core.compile_model(model, device, cfg);
        //// =====================
        auto inputs = compiled.inputs();
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& p = inputs[i];
            std::cerr << "COMPILED IN[" << i << "] name=" << p.get_any_name()
                    << " type=" << p.get_element_type()
                    << " shape=" << p.get_partial_shape() << "\n";
        }
        //// =====================
        auto req = compiled.create_infer_request();

        std::vector<ov::Tensor> outs;
        outs.reserve(steps.size());

        for (const auto& step : steps) {
            for (const auto& kv : step.tensors)
                req.set_tensor(kv.first, kv.second);

            req.infer();
            auto out0 = req.get_output_tensor(0);
            ov::Tensor copy(out0.get_element_type(), out0.get_shape());
            out0.copy_to(copy);
            outs.push_back(std::move(copy));
        }
        return outs;
    }

    void SetUp() override {
        is_report_stages = true;
        const auto& [inType, inputShapes, extendBlockIndices, enableXattn, sinkInput, slidingWindow, additional_config] = GetParam();
        (void)sinkInput;

        init_input_shapes(inputShapes);

        pa_model_ = make_paged_attn_model(inType, enableXattn, 64, 8, /*sink*/false, slidingWindow);

        // Pre-generate the step inputs ONCE so CPU/TEMPLATE see identical tensors.
        // Allocate caches once here, and reuse for both runs by copying initial cache state.
        allocate_caches_for_model(pa_model_, /*block_nums*/4, inType);

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
        // Use model parameter shapes directly (not compiled model) for determinism.
        auto params = model->get_parameters();
        auto kc = params[3];
        auto vc = params[4];

        // Expect rank-4 as constructed above
        auto ps_k = kc->get_partial_shape();
        auto ps_v = vc->get_partial_shape();
        OPENVINO_ASSERT(ps_k.rank().is_static() && ps_k.rank().get_length() == 4);
        OPENVINO_ASSERT(ps_v.rank().is_static() && ps_v.rank().get_length() == 4);

        ps_k[0] = static_cast<int64_t>(block_nums);
        ps_v[0] = static_cast<int64_t>(block_nums);

        // Set explicitly to u8 to mattch CPU & PA precision transformation
        key_cache_init_ = ov::Tensor(ov::element::f32, ps_k.get_shape());
        value_cache_init_ = ov::Tensor(ov::element::f32, ps_v.get_shape());
        std::memset(key_cache_init_.data(), 0, key_cache_init_.get_byte_size());
        std::memset(value_cache_init_.data(), 0, value_cache_init_.get_byte_size());
    }

protected:
    std::shared_ptr<ov::Model> pa_model_;
    ov::Tensor key_cache_init_;
    ov::Tensor value_cache_init_;
    std::vector<StepInputs> steps_;
};

}  // namespace test
}  // namespace ov
