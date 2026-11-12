// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/x64/pass/sparse_attention_kv_cache_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/cpu_opset/common/op/sparse_attention_with_kv_cache.hpp"

using namespace ov;
using ov::intel_cpu::SparseAttentionKVCacheFusion;
using ov::intel_cpu::SparseAttentionWithKVCache;

namespace {

constexpr int64_t kKeyBlockSize = 4;

struct Options {
    bool fused = false;
    bool key_length = false;
    bool state_convert = false;       // Convert between ReadValue and Gather, and between Concat and Assign
    bool shape_of_on_concat = false;  // an extra ShapeOf consumer of the key Concat
    bool extra_cache_reader = false;  // a Result reading the ReadValue of the key state (not a cache-only consumer)
    int64_t query_block_size = 1;
    int64_t concat_axis = 2;
};

// Query [N, 4, L, 16] attends to the cache of key / value [N, 2, L0 + L, 16] kept in two variables.
std::shared_ptr<ov::Model> make_model(const Options& opt) {
    const element::Type et = element::f32;
    auto q = std::make_shared<op::v0::Parameter>(et, PartialShape{-1, 4, -1, 16});
    auto k = std::make_shared<op::v0::Parameter>(et, PartialShape{-1, 2, -1, 16});
    auto v = std::make_shared<op::v0::Parameter>(et, PartialShape{-1, 2, -1, 16});
    auto init = std::make_shared<op::v0::Parameter>(et, PartialShape{-1, 2, -1, 16});
    auto beam_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, 1, -1, 8});
    auto block_count = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, 1, -1});
    auto key_length = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    auto var_k = std::make_shared<op::util::Variable>(op::util::VariableInfo{PartialShape{-1, 2, -1, 16}, et, "pastk"});
    auto var_v = std::make_shared<op::util::Variable>(op::util::VariableInfo{PartialShape{-1, 2, -1, 16}, et, "pastv"});
    std::shared_ptr<Node> past_k = std::make_shared<op::v6::ReadValue>(init, var_k);
    std::shared_ptr<Node> past_v = std::make_shared<op::v6::ReadValue>(init, var_v);
    const auto read_k = past_k;

    ResultVector results;
    Output<Node> attention, present_k, present_v;
    if (opt.state_convert) {
        past_k = std::make_shared<op::v0::Convert>(past_k, et);
        past_v = std::make_shared<op::v0::Convert>(past_v, et);
    }
    if (opt.fused) {
        OutputVector args{q, k, v, block_indices, block_count};
        if (opt.key_length) {
            args.push_back(key_length);
        }
        args.push_back(beam_idx);
        args.push_back(past_k);
        args.push_back(past_v);
        SparseAttentionWithKVCache::Config config;
        config.key_block_size = kKeyBlockSize;
        auto fused = std::make_shared<SparseAttentionWithKVCache>(args, config);
        attention = fused->output(0);
        present_k = fused->output(1);
        present_v = fused->output(2);
        if (opt.shape_of_on_concat) {
            results.push_back(std::make_shared<op::v0::Result>(std::make_shared<op::v3::ShapeOf>(present_k)));
        }
    } else {
        auto axis = op::v0::Constant::create(element::i32, {1}, {0});
        auto gather_k = std::make_shared<op::v8::Gather>(past_k, beam_idx, axis);
        auto gather_v = std::make_shared<op::v8::Gather>(past_v, beam_idx, axis);
        auto concat_k = std::make_shared<op::v0::Concat>(OutputVector{gather_k, k}, opt.concat_axis);
        auto concat_v = std::make_shared<op::v0::Concat>(OutputVector{gather_v, v}, opt.concat_axis);
        std::shared_ptr<Node> sparse;
        if (opt.key_length) {
            sparse = std::make_shared<op::v17::SparseAttention>(q,
                                                                concat_k,
                                                                concat_v,
                                                                block_indices,
                                                                block_count,
                                                                key_length,
                                                                kKeyBlockSize,
                                                                opt.query_block_size);
        } else {
            sparse = std::make_shared<op::v17::SparseAttention>(q,
                                                                concat_k,
                                                                concat_v,
                                                                block_indices,
                                                                block_count,
                                                                kKeyBlockSize,
                                                                opt.query_block_size);
        }
        attention = sparse->output(0);
        present_k = concat_k;
        present_v = concat_v;
        if (opt.shape_of_on_concat) {
            results.push_back(std::make_shared<op::v0::Result>(std::make_shared<op::v3::ShapeOf>(concat_k)));
        }
    }
    if (opt.extra_cache_reader) {
        results.push_back(std::make_shared<op::v0::Result>(read_k));
    }
    Output<Node> assign_k_input = present_k;
    Output<Node> assign_v_input = present_v;
    if (opt.state_convert) {
        assign_k_input = std::make_shared<op::v0::Convert>(present_k, et);
        assign_v_input = std::make_shared<op::v0::Convert>(present_v, et);
    }
    auto assign_k = std::make_shared<op::v6::Assign>(assign_k_input, var_k);
    auto assign_v = std::make_shared<op::v6::Assign>(assign_v_input, var_v);
    results.insert(results.begin(), std::make_shared<op::v0::Result>(attention));
    ParameterVector params{q, k, v, init, beam_idx, block_indices, block_count};
    if (opt.key_length) {
        params.push_back(key_length);
    }
    return std::make_shared<ov::Model>(results, SinkVector{assign_k, assign_v}, params, "SparseKV");
}

void expect_fused(const std::shared_ptr<ov::Model>& model, bool fused) {
    size_t fused_count = 0;
    size_t sparse_count = 0;
    for (const auto& op : model->get_ops()) {
        fused_count += ov::is_type<SparseAttentionWithKVCache>(op) ? 1 : 0;
        sparse_count += ov::is_type<op::v17::SparseAttention>(op) ? 1 : 0;
    }
    EXPECT_EQ(fused_count, fused ? 1U : 0U);
    EXPECT_EQ(sparse_count, fused ? 0U : 1U);
}

}  // namespace

class SparseAttentionKVCacheFusionTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        disable_rt_info_check();
        manager.register_pass<SparseAttentionKVCacheFusion>();
    }
};

TEST_F(SparseAttentionKVCacheFusionTest, Fuses) {
    Options opt;
    model = make_model(opt);
    opt.fused = true;
    model_ref = make_model(opt);
}

TEST_F(SparseAttentionKVCacheFusionTest, FusesWithKeyLengthAndConverts) {
    Options opt;
    opt.key_length = true;
    opt.state_convert = true;
    model = make_model(opt);
    opt.fused = true;
    model_ref = make_model(opt);
}

TEST_F(SparseAttentionKVCacheFusionTest, FusesAndReroutesShapeOfOfConcat) {
    Options opt;
    opt.shape_of_on_concat = true;
    model = make_model(opt);
    opt.fused = true;
    model_ref = make_model(opt);
}

TEST_F(SparseAttentionKVCacheFusionTest, NoFusionWhenCacheHasOtherReader) {
    Options opt;
    opt.extra_cache_reader = true;
    model = make_model(opt);
    model_ref = model->clone();
}

TEST_F(SparseAttentionKVCacheFusionTest, NoFusionForQueryBlockSizeAboveOne) {
    Options opt;
    opt.query_block_size = 2;
    model = make_model(opt);
    model_ref = model->clone();
}

TEST(SparseAttentionKVCacheFusionStructure, SinksAreFedByTheFusedOp) {
    auto model = make_model({});
    ov::pass::Manager manager;
    manager.register_pass<SparseAttentionKVCacheFusion>();
    manager.run_passes(model);
    expect_fused(model, true);
    for (const auto& sink : model->get_sinks()) {
        EXPECT_TRUE(ov::is_type<SparseAttentionWithKVCache>(sink->get_input_node_shared_ptr(0)));
    }
}
