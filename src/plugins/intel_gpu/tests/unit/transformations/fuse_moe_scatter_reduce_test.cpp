// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_moe_scatter_reduce.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_scatter_reduce.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov::test::intel_gpu {

namespace {

constexpr int64_t hidden_size = 16;

enum class LutSource {
    ParamI32,
    ParamI64,
    Argsort,          // i64 argsort of a permutation, as exported for the inverse permutation
    ConvertedArgsort  // the argsort converted to i32
};

struct ModelConfig {
    LutSource lut_source = LutSource::Argsort;
    int64_t reduce_axis = 1;
    bool weights_first = false;  // Multiply(weights, rows)
    bool keep_dims = false;
    bool extra_mul_consumer = false;
    int64_t top_k = 2;
    ov::Dimension weights_tokens = ov::Dimension::dynamic();
    ov::Dimension lut_rows = ov::Dimension::dynamic();
};

ov::ParameterVector make_params(const ModelConfig& cfg) {
    const auto lut_type = cfg.lut_source == LutSource::ParamI32 ? ov::element::i32 : ov::element::i64;
    return {std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, hidden_size}),
            std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{cfg.weights_tokens, cfg.top_k}),
            std::make_shared<ov::op::v0::Parameter>(lut_type, ov::PartialShape{cfg.lut_rows})};
}

ov::Output<ov::Node> make_lut(const ModelConfig& cfg, const ov::Output<ov::Node>& lut_param) {
    if (cfg.lut_source == LutSource::ParamI32 || cfg.lut_source == LutSource::ParamI64) {
        return lut_param;
    }
    auto len = std::make_shared<ov::op::v0::Squeeze>(std::make_shared<ov::op::v3::ShapeOf>(lut_param));
    ov::Output<ov::Node> lut = std::make_shared<ov::op::v11::TopK>(lut_param,
                                                                   len,
                                                                   -1,
                                                                   ov::op::v11::TopK::Mode::MIN,
                                                                   ov::op::v11::TopK::SortType::SORT_VALUES,
                                                                   ov::element::i64)
                                   ->output(1);
    if (cfg.lut_source == LutSource::ConvertedArgsort) {
        lut = std::make_shared<ov::op::v0::Convert>(lut, ov::element::i32);
    }
    return lut;
}

std::shared_ptr<ov::Model> build_model(const ModelConfig& cfg) {
    auto params = make_params(cfg);
    const auto& [data, weights, lut_param] = std::tie(params[0], params[1], params[2]);

    auto lut = make_lut(cfg, lut_param);
    auto rows = std::make_shared<ov::op::v8::Gather>(data, lut, ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
    auto flat_weights = std::make_shared<ov::op::v1::Reshape>(weights, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
    auto col_weights = std::make_shared<ov::op::v0::Unsqueeze>(flat_weights, ov::op::v0::Constant::create(ov::element::i64, {}, {-1}));
    auto weighted = cfg.weights_first ? std::make_shared<ov::op::v1::Multiply>(col_weights, rows) : std::make_shared<ov::op::v1::Multiply>(rows, col_weights);
    auto split = std::make_shared<ov::op::v1::Reshape>(weighted,
                                                       ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{-1, cfg.top_k, hidden_size}),
                                                       false);
    auto reduce = std::make_shared<ov::op::v1::ReduceSum>(split, ov::op::v0::Constant::create(ov::element::i64, {1}, {cfg.reduce_axis}), cfg.keep_dims);
    reduce->set_friendly_name("moe_out");

    ov::OutputVector outputs{reduce->output(0)};
    if (cfg.extra_mul_consumer) {
        outputs.push_back(weighted);
    }
    return std::make_shared<ov::Model>(outputs, params);
}

std::shared_ptr<ov::Model> build_model_ref(const ModelConfig& cfg) {
    auto params = make_params(cfg);
    auto combine = std::make_shared<ov::intel_gpu::op::MoEScatterReduce>(params[0], params[1], make_lut(cfg, params[2]));
    combine->set_friendly_name("moe_out");
    return std::make_shared<ov::Model>(ov::OutputVector{combine->output(0)}, params);
}

}  // namespace

using FuseMoEScatterReduceParams = std::tuple<LutSource, int64_t, bool>;  // row_lut source, reduce axis, weights first

class FuseMoEScatterReduceTest : public TransformationTestsF, public WithParamInterface<FuseMoEScatterReduceParams> {
public:
    static std::string get_test_case_name(const TestParamInfo<FuseMoEScatterReduceParams>& info) {
        const auto& [lut_source, reduce_axis, weights_first] = info.param;
        static const char* const lut_names[] = {"LutParamI32", "LutParamI64", "LutArgsort", "LutConvertedArgsort"};
        return std::string(lut_names[static_cast<int>(lut_source)]) + "_axis" + (reduce_axis < 0 ? "m" : "") +
               std::to_string(std::abs(reduce_axis)) + (weights_first ? "_WeightsFirst" : "_RowsFirst");
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        ModelConfig cfg;
        std::tie(cfg.lut_source, cfg.reduce_axis, cfg.weights_first) = GetParam();
        model = build_model(cfg);
        model_ref = build_model_ref(cfg);
        manager.register_pass<FuseMoEScatterReduce>();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
};

TEST_P(FuseMoEScatterReduceTest, Fused) {}

INSTANTIATE_TEST_SUITE_P(smoke,
                         FuseMoEScatterReduceTest,
                         Combine(Values(LutSource::ParamI32, LutSource::ParamI64, LutSource::Argsort, LutSource::ConvertedArgsort),
                                 Values(int64_t{1}, int64_t{-2}),
                                 Values(false, true)),
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

TEST_F(TransformationTestsF, FuseMoEScatterReduce_WeightedRowsHaveOtherConsumer) {
    ModelConfig cfg;
    cfg.extra_mul_consumer = true;
    model = build_model(cfg);
    manager.register_pass<FuseMoEScatterReduce>();
}

// With top_k = 1 the weights must have the rows of row_lut
TEST_F(TransformationTestsF, FuseMoEScatterReduce_TopK1SameRows) {
    ModelConfig cfg;
    cfg.top_k = 1;
    cfg.weights_tokens = 4;
    cfg.lut_rows = 4;
    model = build_model(cfg);
    model_ref = build_model_ref(cfg);
    manager.register_pass<FuseMoEScatterReduce>();
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

// The Multiply broadcasts the one weight to all the rows
TEST_F(TransformationTestsF, FuseMoEScatterReduce_TopK1BroadcastWeight) {
    ModelConfig cfg;
    cfg.top_k = 1;
    cfg.weights_tokens = 1;
    model = build_model(cfg);
    manager.register_pass<FuseMoEScatterReduce>();
}

}  // namespace ov::test::intel_gpu
