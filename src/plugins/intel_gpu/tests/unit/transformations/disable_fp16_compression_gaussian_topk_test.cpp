// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

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
const std::string name_threshold_mean = "threshold_mean";
const std::string name_deviation = "deviation";
const std::string name_squared_deviation = "squared_deviation";
const std::string name_variance = "variance";
const std::string name_std_dev = "std_dev";
const std::string name_scaled_std_dev = "scaled_std_dev";
const std::string name_threshold = "threshold";
const std::string name_sparsified = "sparsified";
const std::string name_relu = "relu";

// SeparateMeans is a positive variant. Each negative variant but the last still matches the pattern and is rejected by one
// callback check; the last one is rejected by the pattern itself, as its data input is bound once.
enum class Variant {
    GaussianTopK,
    SeparateMeans,                  // threshold uses its own, equivalent ReduceMean (SharedOpOptimization has not run)
    NonSquaredDeviation,            // Multiply(x - mean_1(x), deviation): not a square
    ThresholdFromOtherMean,         // threshold built on mean_1(x) instead of the deviation's mean
    ThresholdMeanKeepDimsMismatch,  // threshold built on a mean with keep_dims=false
    MismatchedReduceAxes,           // variance reduced over axis 1, mean over the last axis
    ThresholdMeanOfOtherInput,      // threshold built on the mean of another tensor
};

// relu(x - (mean(x) + k * sqrt(mean((x - mean(x))^2)))), as emitted by the PyTorch frontend
// for torch.std-based Gaussian top-k activation sparsity (e.g. Gemma-3n MLP).
std::shared_ptr<ov::Model> create_model(Variant variant, bool mean_first_in_threshold = true) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 8});
    ov::ParameterVector parameters{x};
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
    if (variant == Variant::SeparateMeans)
        threshold_mean = std::make_shared<ov::op::v1::ReduceMean>(x, axes, true);
    else if (variant == Variant::ThresholdFromOtherMean)
        threshold_mean = std::make_shared<ov::op::v1::ReduceMean>(x, axis_1, true);
    else if (variant == Variant::ThresholdMeanKeepDimsMismatch)
        threshold_mean = std::make_shared<ov::op::v1::ReduceMean>(x, axes, false);
    else if (variant == Variant::ThresholdMeanOfOtherInput) {
        auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 8});
        parameters.push_back(y);
        threshold_mean = std::make_shared<ov::op::v1::ReduceMean>(y, axes, true);
    }
    if (threshold_mean != mean)
        threshold_mean->set_friendly_name(name_threshold_mean);
    auto threshold = mean_first_in_threshold ? std::make_shared<ov::op::v1::Add>(threshold_mean, scaled_std_dev)
                                             : std::make_shared<ov::op::v1::Add>(scaled_std_dev, threshold_mean);
    threshold->set_friendly_name(name_threshold);
    auto sparsified = std::make_shared<ov::op::v1::Subtract>(x, threshold);
    sparsified->set_friendly_name(name_sparsified);
    auto relu = std::make_shared<ov::op::v0::Relu>(sparsified);
    relu->set_friendly_name(name_relu);

    return std::make_shared<ov::Model>(ov::OutputVector{relu}, parameters);
}

// Runs the pass alone: the marks checked here are the ones the pass itself sets.
void run_test(const std::shared_ptr<ov::Model>& model,
              const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGaussianTopK>();
    manager.run_passes(model);

    size_t matched_count = 0;
    for (const auto& op : model->get_ops()) {
        auto it = expected_fp16_disabled_status.find(op->get_friendly_name());
        if (it == expected_fp16_disabled_status.end())
            continue;
        ++matched_count;
        if (it->second) {
            ASSERT_TRUE(ov::is_conversion_disabled(op, ov::element::f16))
                << "FP16 compression is not disabled for node: " << op->get_friendly_name();
        } else {
            ASSERT_FALSE(ov::is_conversion_disabled(op, ov::element::f16))
                << "FP16 compression is unexpectedly disabled for node: " << op->get_friendly_name();
        }
    }
    ASSERT_EQ(matched_count, expected_fp16_disabled_status.size()) << "Some expected nodes are missing in the model";
}
}  // namespace

TEST(TransformationTests, DisableFP16CompForGaussianTopK_Positive) {
    // The pass itself marks the squared deviation and the nodes computed from it. mean and deviation are not marked
    // here: ConvertPrecision propagates the mark up to them (see DisableFP16CompForGaussianTopK_KeepPrecisionSensitiveInFP32).
    std::unordered_map<std::string, bool> expected_status = {
        {name_mean, false},
        {name_deviation, false},
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

TEST(TransformationTests, DisableFP16CompForGaussianTopK_SeparateMeans_Positive) {
    std::unordered_map<std::string, bool> expected_status = {
        {name_mean, false},
        {name_threshold_mean, false},
        {name_deviation, false},
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
        run_test(create_model(Variant::SeparateMeans, mean_first), expected_status);
    }
}

// ConvertPrecision is registered as in the GPU plugin: the mark set by the pass propagates up to mean and deviation.
TEST(TransformationTests, DisableFP16CompForGaussianTopK_KeepPrecisionSensitiveInFP32) {
    for (auto variant : {Variant::GaussianTopK, Variant::SeparateMeans}) {
        SCOPED_TRACE(variant == Variant::GaussianTopK ? "shared mean" : "separate means");
        auto model = create_model(variant);

        ov::pass::Manager manager;
        manager.register_pass<DisableFP16CompForGaussianTopK>();
        precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
        manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                          type_to_fuse_map{},
                                                          true,   // keep_precision_sensitive_in_fp32
                                                          false,  // convert_input_output_precision
                                                          true);  // store_original_precision_as_rt_attribute
        manager.run_passes(model);

        std::unordered_set<std::string> expected_fp32_nodes =
            {name_mean, name_deviation, name_squared_deviation, name_variance, name_std_dev, name_scaled_std_dev, name_threshold};
        if (variant == Variant::SeparateMeans)
            expected_fp32_nodes.insert(name_threshold_mean);

        size_t matched_count = 0;
        for (const auto& op : model->get_ops()) {
            if (expected_fp32_nodes.count(op->get_friendly_name()) == 0)
                continue;
            ++matched_count;
            ASSERT_EQ(op->get_output_element_type(0), ov::element::f32) << "Node is converted to FP16: " << op->get_friendly_name();
        }
        ASSERT_EQ(matched_count, expected_fp32_nodes.size()) << "Some expected nodes are missing in the model";
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
        {name_threshold_mean, false},
        {name_deviation, false},
        {name_squared_deviation, false},
        {name_variance, false},
        {name_threshold, false},
    };
    run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompForGaussianTopK_ThresholdMeanKeepDimsMismatch_NoOp) {
    auto model = create_model(Variant::ThresholdMeanKeepDimsMismatch);
    std::unordered_map<std::string, bool> expected_status = {
        {name_mean, false},
        {name_threshold_mean, false},
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

TEST(TransformationTests, DisableFP16CompForGaussianTopK_ThresholdMeanOfOtherInput_NoOp) {
    auto model = create_model(Variant::ThresholdMeanOfOtherInput);
    std::unordered_map<std::string, bool> expected_status = {
        {name_mean, false},
        {name_threshold_mean, false},
        {name_deviation, false},
        {name_squared_deviation, false},
        {name_variance, false},
        {name_threshold, false},
    };
    run_test(model, expected_status);
}
