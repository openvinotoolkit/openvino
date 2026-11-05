// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_moe_scatter_reduce.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_expert_sort.hpp"
#include "intel_gpu/op/moe_scatter_reduce.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov::test::intel_gpu {

namespace {

constexpr int64_t hidden_size = 16;
constexpr int64_t top_k = 2;
constexpr size_t num_experts = 8;

struct ModelConfig {
    bool lut_converts = true;        // row_lut reaches the Gather through i64 -> i32 Converts, as left by FuseMoEExpertSort
    int64_t reduce_axis = 1;
    bool weights_first = false;      // Multiply(weights, rows)
    bool keep_dims = false;
    bool lut_from_sort = true;       // row_lut is a Parameter otherwise
    bool extra_mul_consumer = false;
};

ov::ParameterVector make_params() {
    return {std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, hidden_size}),
            std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, top_k}),
            std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, top_k}),
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1})};
}

std::shared_ptr<ov::Model> build_model(const ModelConfig& cfg) {
    auto params = make_params();
    const auto& [data, weights, topk_ids, lut_param] = std::tie(params[0], params[1], params[2], params[3]);

    auto sort = std::make_shared<ov::intel_gpu::op::MoEExpertSort>(topk_ids, num_experts);
    ov::Output<ov::Node> lut = cfg.lut_from_sort ? sort->output(2) : lut_param->output(0);
    if (cfg.lut_converts) {
        lut = std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v0::Convert>(lut, ov::element::i64), ov::element::i32);
    }
    auto rows = std::make_shared<ov::op::v8::Gather>(data, lut, ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
    auto flat_weights = std::make_shared<ov::op::v1::Reshape>(weights, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
    auto col_weights = std::make_shared<ov::op::v0::Unsqueeze>(flat_weights, ov::op::v0::Constant::create(ov::element::i64, {}, {-1}));
    auto weighted = cfg.weights_first ? std::make_shared<ov::op::v1::Multiply>(col_weights, rows) : std::make_shared<ov::op::v1::Multiply>(rows, col_weights);
    auto split = std::make_shared<ov::op::v1::Reshape>(weighted,
                                                       ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{-1, top_k, hidden_size}),
                                                       false);
    auto reduce = std::make_shared<ov::op::v1::ReduceSum>(split, ov::op::v0::Constant::create(ov::element::i64, {1}, {cfg.reduce_axis}), cfg.keep_dims);
    reduce->set_friendly_name("moe_out");

    ov::OutputVector outputs{reduce->output(0), sort->output(1)};
    if (cfg.extra_mul_consumer) {
        outputs.push_back(weighted);
    }
    return std::make_shared<ov::Model>(outputs, params);
}

std::shared_ptr<ov::Model> build_model_ref() {
    auto params = make_params();
    auto sort = std::make_shared<ov::intel_gpu::op::MoEExpertSort>(params[2], num_experts);
    auto combine = std::make_shared<ov::intel_gpu::op::MoEScatterReduce>(params[0], params[1], sort->output(2));
    combine->set_friendly_name("moe_out");
    return std::make_shared<ov::Model>(ov::OutputVector{combine->output(0), sort->output(1)}, params);
}

}  // namespace

using FuseMoEScatterReduceParams = std::tuple<bool, int64_t, bool>;  // lut Converts, reduce axis, weights first

class FuseMoEScatterReduceTest : public TransformationTestsF, public WithParamInterface<FuseMoEScatterReduceParams> {
public:
    static std::string get_test_case_name(const TestParamInfo<FuseMoEScatterReduceParams>& info) {
        const auto& [lut_converts, reduce_axis, weights_first] = info.param;
        return std::string(lut_converts ? "LutConverts" : "LutDirect") + "_axis" + (reduce_axis < 0 ? "m" : "") +
               std::to_string(std::abs(reduce_axis)) + (weights_first ? "_WeightsFirst" : "_RowsFirst");
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        ModelConfig cfg;
        std::tie(cfg.lut_converts, cfg.reduce_axis, cfg.weights_first) = GetParam();
        model = build_model(cfg);
        model_ref = build_model_ref();
        manager.register_pass<FuseMoEScatterReduce>();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
};

TEST_P(FuseMoEScatterReduceTest, Fused) {}

INSTANTIATE_TEST_SUITE_P(smoke,
                         FuseMoEScatterReduceTest,
                         Combine(Values(true, false), Values(int64_t{1}, int64_t{-2}), Values(false, true)),
                         FuseMoEScatterReduceTest::get_test_case_name);

TEST_F(TransformationTestsF, FuseMoEScatterReduce_KeepDims) {
    ModelConfig cfg;
    cfg.keep_dims = true;
    model = build_model(cfg);
    manager.register_pass<FuseMoEScatterReduce>();
}

TEST_F(TransformationTestsF, FuseMoEScatterReduce_ReduceOverHidden) {
    ModelConfig cfg;
    cfg.reduce_axis = 2;
    model = build_model(cfg);
    manager.register_pass<FuseMoEScatterReduce>();
}

TEST_F(TransformationTestsF, FuseMoEScatterReduce_LutNotFromExpertSort) {
    ModelConfig cfg;
    cfg.lut_from_sort = false;
    model = build_model(cfg);
    manager.register_pass<FuseMoEScatterReduce>();
}

TEST_F(TransformationTestsF, FuseMoEScatterReduce_WeightedRowsHaveOtherConsumer) {
    ModelConfig cfg;
    cfg.extra_mul_consumer = true;
    model = build_model(cfg);
    manager.register_pass<FuseMoEScatterReduce>();
}

}  // namespace ov::test::intel_gpu
