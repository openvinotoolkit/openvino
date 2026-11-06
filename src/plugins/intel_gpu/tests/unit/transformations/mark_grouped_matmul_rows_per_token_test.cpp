// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/mark_grouped_matmul_rows_per_token.hpp"

#include <gtest/gtest.h>

#include "intel_gpu/op/moe_expert_sort.hpp"
#include "intel_gpu/op/moe_router_fused.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/grouped_matmul_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov::test::intel_gpu {

namespace {

constexpr int64_t top_k = 2;
constexpr int64_t num_experts = 8;
constexpr int64_t hidden_in = 16;
constexpr int64_t hidden_out = 32;

enum class IdsSource { TopK, TopKAxis1, RouterFused, TopKAxis0, TopKDynamicK, Parameter };
enum class GroupedMatMulKind { Plain, Compressed, CompressedZp };

struct ModelConfig {
    IdsSource ids_source = IdsSource::TopK;
    bool ids_convert = false;
    bool offsets_convert = false;
    GroupedMatMulKind kind = GroupedMatMulKind::Plain;
};

// Returns the rows-per-token key set on the GroupedMatMul by the pass, or -1 if it is not set.
int64_t run_and_get_rows_per_token(const ModelConfig& cfg) {
    using namespace ov::op;
    auto logits = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, num_experts});
    ov::ParameterVector params{logits};
    ov::Output<ov::Node> ids;
    const auto k = v0::Constant::create(element::i64, {}, {top_k});
    const auto max = v11::TopK::Mode::MAX;
    const auto by_value = v11::TopK::SortType::SORT_VALUES;
    switch (cfg.ids_source) {
    case IdsSource::TopK:
        ids = std::make_shared<v11::TopK>(logits, k, -1, max, by_value, element::i64)->output(1);
        break;
    case IdsSource::TopKAxis1:
        ids = std::make_shared<v11::TopK>(logits, k, 1, max, by_value, element::i64)->output(1);
        break;
    case IdsSource::TopKAxis0:
        ids = std::make_shared<v11::TopK>(logits, k, 0, max, by_value, element::i64)->output(1);
        break;
    case IdsSource::TopKDynamicK: {
        auto dynamic_k = std::make_shared<v0::Parameter>(element::i64, PartialShape{});
        params.push_back(dynamic_k);
        ids = std::make_shared<v11::TopK>(logits, dynamic_k, -1, max, by_value, element::i64)->output(1);
        break;
    }
    case IdsSource::RouterFused: {
        ov::intel_gpu::op::MoERouterFused::Config router_cfg;
        router_cfg.num_expert = num_experts;
        router_cfg.top_k = top_k;
        ids = std::make_shared<ov::intel_gpu::op::MoERouterFused>(ov::OutputVector{logits}, router_cfg)->output(1);
        break;
    }
    case IdsSource::Parameter: {
        auto p = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1, top_k});
        params.push_back(p);
        ids = p;
        break;
    }
    }
    if (cfg.ids_convert) {
        ids = std::make_shared<v0::Convert>(ids, element::i32);
    }
    auto sort = std::make_shared<ov::intel_gpu::op::MoEExpertSort>(ids, num_experts);
    ov::Output<ov::Node> offsets = sort->output(1);
    if (cfg.offsets_convert) {
        offsets = std::make_shared<v0::Convert>(offsets, element::i64);
    }

    auto mat_a = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, hidden_in});
    params.push_back(mat_a);
    std::shared_ptr<ov::Node> gmm;
    if (cfg.kind == GroupedMatMulKind::Plain) {
        auto mat_b = std::make_shared<v0::Parameter>(element::f16, PartialShape{num_experts, hidden_out, hidden_in});
        params.push_back(mat_b);
        gmm = std::make_shared<v17::GroupedMatMul>(mat_a, mat_b, offsets);
    } else {
        auto mat_b = v0::Constant::create(element::u4, {num_experts, hidden_out, hidden_in}, {1});
        auto scale = v0::Constant::create(element::f16, {num_experts, hidden_out}, {1});
        if (cfg.kind == GroupedMatMulKind::Compressed) {
            gmm = std::make_shared<ov::op::internal::GroupedMatMulCompressed>(mat_a, mat_b, offsets, scale);
        } else {
            auto zp = v0::Constant::create(element::u4, {num_experts, hidden_out}, {8});
            gmm = std::make_shared<ov::op::internal::GroupedMatMulCompressed>(mat_a, mat_b, offsets, scale, zp);
        }
    }
    auto model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, params);

    ov::pass::Manager manager;
    manager.register_pass<MarkGroupedMatMulRowsPerToken>();
    manager.run_passes(model);

    const auto& rt = gmm->get_rt_info();
    const auto it = rt.find(grouped_matmul_rows_per_token_key);
    return it == rt.end() ? -1 : it->second.as<int64_t>();
}

std::string ids_source_name(IdsSource s) {
    switch (s) {
    case IdsSource::TopK:
        return "TopK";
    case IdsSource::TopKAxis1:
        return "TopKAxis1";
    case IdsSource::RouterFused:
        return "RouterFused";
    case IdsSource::TopKAxis0:
        return "TopKAxis0";
    case IdsSource::TopKDynamicK:
        return "TopKDynamicK";
    case IdsSource::Parameter:
        return "Parameter";
    }
    return "";
}

std::string kind_name(GroupedMatMulKind k) {
    if (k == GroupedMatMulKind::Plain)
        return "Plain";
    return k == GroupedMatMulKind::Compressed ? "Compressed" : "CompressedZp";
}

}  // namespace

// ids source, ids Convert, offsets Convert, GroupedMatMul kind
using MarkRowsPerTokenParams = std::tuple<IdsSource, bool, bool, GroupedMatMulKind>;

class MarkGroupedMatMulRowsPerTokenTest : public TestWithParam<MarkRowsPerTokenParams> {
public:
    static std::string get_test_case_name(const TestParamInfo<MarkRowsPerTokenParams>& info) {
        const auto& [ids_source, ids_convert, offsets_convert, kind] = info.param;
        return ids_source_name(ids_source) + (ids_convert ? "_IdsConvert" : "") +
               (offsets_convert ? "_OffsetsConvert" : "") + "_" + kind_name(kind);
    }
};

TEST_P(MarkGroupedMatMulRowsPerTokenTest, Marked) {
    ModelConfig cfg;
    std::tie(cfg.ids_source, cfg.ids_convert, cfg.offsets_convert, cfg.kind) = GetParam();
    EXPECT_EQ(run_and_get_rows_per_token(cfg), top_k);
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         MarkGroupedMatMulRowsPerTokenTest,
                         Combine(Values(IdsSource::TopK, IdsSource::TopKAxis1, IdsSource::RouterFused),
                                 Values(false, true),
                                 Values(false, true),
                                 Values(GroupedMatMulKind::Plain,
                                        GroupedMatMulKind::Compressed,
                                        GroupedMatMulKind::CompressedZp)),
                         MarkGroupedMatMulRowsPerTokenTest::get_test_case_name);

class MarkGroupedMatMulRowsPerTokenNegativeTest : public TestWithParam<IdsSource> {};

TEST_P(MarkGroupedMatMulRowsPerTokenNegativeTest, NotMarked) {
    ModelConfig cfg;
    cfg.ids_source = GetParam();
    EXPECT_EQ(run_and_get_rows_per_token(cfg), -1);
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         MarkGroupedMatMulRowsPerTokenNegativeTest,
                         Values(IdsSource::TopKAxis0, IdsSource::TopKDynamicK, IdsSource::Parameter),
                         [](const TestParamInfo<IdsSource>& info) {
                             return ids_source_name(info.param);
                         });

}  // namespace ov::test::intel_gpu
