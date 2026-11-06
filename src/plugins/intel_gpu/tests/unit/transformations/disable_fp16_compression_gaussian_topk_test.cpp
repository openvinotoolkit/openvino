// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/rt_info/disable_precision_conversion.hpp>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "plugin/transformations/disable_fp16_comp_gaussian_topk.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {
const std::string name_mean = "mean";
const std::string name_deviation = "deviation";
const std::string name_squared_deviation = "squared_deviation";
const std::string name_variance = "variance";
const std::string name_std_dev = "std_dev";
const std::string name_scaled_std_dev = "scaled_std_dev";
const std::string name_threshold = "threshold";
const std::string name_sparsified = "sparsified";
const std::string name_relu = "relu";

// Each negative variant still matches the pattern and is rejected by one callback check.
enum class Variant {
    GaussianTopK,
    NonSquaredDeviation,     // Multiply(x - mean_1(x), deviation): not a square
    ThresholdFromOtherMean,  // threshold built on mean_1(x) instead of the deviation's mean
    MismatchedReduceAxes,    // variance reduced over axis 1, mean over the last axis
};

// relu(x - (mean(x) + k * sqrt(mean((x - mean(x))^2)))), as emitted by the PyTorch frontend
// for torch.std-based Gaussian top-k activation sparsity (e.g. Gemma-3n MLP).
std::shared_ptr<ov::Model> create_model(Variant variant, bool mean_first_in_threshold = true) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 64});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(x, axes, true);
    mean->set_friendly_name(name_mean);
    auto deviation = std::make_shared<ov::op::v1::Subtract>(x, mean);
    deviation->set_friendly_name(name_deviation);
    std::shared_ptr<ov::Node> squared_deviation = std::make_shared<ov::op::v1::Multiply>(deviation, deviation);
    if (variant == Variant::NonSquaredDeviation) {
        auto other_deviation = std::make_shared<ov::op::v1::Subtract>(x, std::make_shared<ov::op::v1::ReduceMean>(x, axis_1, true));
        squared_deviation = std::make_shared<ov::op::v1::Multiply>(other_deviation, deviation);
    }
    squared_deviation->set_friendly_name(name_squared_deviation);
    auto variance = std::make_shared<ov::op::v1::ReduceMean>(squared_deviation, variant == Variant::MismatchedReduceAxes ? axis_1 : axes, true);
    variance->set_friendly_name(name_variance);
    auto std_dev = std::make_shared<ov::op::v0::Sqrt>(variance);
    std_dev->set_friendly_name(name_std_dev);
    auto k = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.6448536f});
    auto scaled_std_dev = std::make_shared<ov::op::v1::Multiply>(std_dev, k);
    scaled_std_dev->set_friendly_name(name_scaled_std_dev);

    std::shared_ptr<ov::Node> threshold_mean = mean;
    if (variant == Variant::ThresholdFromOtherMean)
        threshold_mean = std::make_shared<ov::op::v1::ReduceMean>(x, axis_1, true);
    auto threshold = mean_first_in_threshold ? std::make_shared<ov::op::v1::Add>(threshold_mean, scaled_std_dev)
                                             : std::make_shared<ov::op::v1::Add>(scaled_std_dev, threshold_mean);
    threshold->set_friendly_name(name_threshold);
    auto sparsified = std::make_shared<ov::op::v1::Subtract>(x, threshold);
    sparsified->set_friendly_name(name_sparsified);
    auto relu = std::make_shared<ov::op::v0::Relu>(sparsified);
    relu->set_friendly_name(name_relu);

    return std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{x});
}

void run_test(const std::shared_ptr<ov::Model>& model,
              const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGaussianTopK>();

    precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map);

    manager.run_passes(model);

    for (const auto& op : model->get_ops()) {
        auto it = expected_fp16_disabled_status.find(op->get_friendly_name());
        if (it == expected_fp16_disabled_status.end())
            continue;
        if (it->second) {
            ASSERT_TRUE(ov::is_conversion_disabled(op, ov::element::f16))
                << "FP16 compression is not disabled for node: " << op->get_friendly_name();
        } else {
            ASSERT_FALSE(ov::is_conversion_disabled(op, ov::element::f16))
                << "FP16 compression is unexpectedly disabled for node: " << op->get_friendly_name();
        }
    }
}
}  // namespace

TEST(TransformationTests, DisableFP16CompForGaussianTopK_Positive) {
    // Only the threshold computation is kept in FP32; the sparsified output is not.
    std::unordered_map<std::string, bool> expected_status = {
        {name_mean, true},
        {name_deviation, true},
        {name_squared_deviation, true},
        {name_variance, true},
        {name_std_dev, true},
        {name_scaled_std_dev, true},
        {name_threshold, true},
        {name_sparsified, false},
        {name_relu, false},
    };
    for (bool mean_first : {true, false}) {
        SCOPED_TRACE(mean_first ? "Add(mean, k * std)" : "Add(k * std, mean)");
        run_test(create_model(Variant::GaussianTopK, mean_first), expected_status);
    }
}

TEST(TransformationTests, DisableFP16CompForGaussianTopK_NonSquaredDeviation_NoOp) {
    auto model = create_model(Variant::NonSquaredDeviation);
    std::unordered_map<std::string, bool> expected_status = {
        {name_mean, false},
        {name_deviation, false},
        {name_squared_deviation, false},
        {name_variance, false},
        {name_threshold, false},
    };
    run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompForGaussianTopK_ThresholdFromOtherMean_NoOp) {
    auto model = create_model(Variant::ThresholdFromOtherMean);
    std::unordered_map<std::string, bool> expected_status = {
        {name_mean, false},
        {name_deviation, false},
        {name_squared_deviation, false},
        {name_variance, false},
        {name_threshold, false},
    };
    run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompForGaussianTopK_MismatchedReduceAxes_NoOp) {
    auto model = create_model(Variant::MismatchedReduceAxes);
    std::unordered_map<std::string, bool> expected_status = {
        {name_mean, false},
        {name_deviation, false},
        {name_squared_deviation, false},
        {name_variance, false},
        {name_threshold, false},
    };
    run_test(model, expected_status);
}
