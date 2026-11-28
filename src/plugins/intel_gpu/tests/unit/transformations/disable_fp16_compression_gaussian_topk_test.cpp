// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <string>
#include <transformations/convert_precision.hpp>
#include <transformations/rt_info/disable_constant_folding.hpp>
#include <transformations/rt_info/disable_precision_conversion.hpp>
#include <tuple>
#include <unordered_set>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/greater.hpp"
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
// With is_reference the nodes marked by the pass carry the FP16 conversion mark.
std::shared_ptr<ov::Model> create_model(Variant variant, bool mean_first_in_threshold = true, bool is_reference = false) {
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

    if (is_reference) {
        for (const auto& node : ov::NodeVector{squared_deviation, variance, std_dev, scaled_std_dev, threshold})
            ov::disable_conversion(node, ov::element::f16);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{relu}, parameters);
}

// The model after DisableFP16CompForGaussianTopK and ConvertPrecision as registered in the GPU plugin: the FP32 mark
// spreads over the whole subgraph and every consumer of x reads it through a Convert.
std::shared_ptr<ov::Model> create_mixed_precision_model(Variant variant) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 8});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});

    auto keep_f32 = [](std::shared_ptr<ov::Node> node) {
        ov::disable_conversion(node, ov::element::f16);
        return node;
    };
    auto x_f32 = [&]() {
        auto convert = std::make_shared<ov::op::v0::Convert>(x, ov::element::f32);
        ov::disable_constant_folding(convert);
        return keep_f32(convert);
    };

    auto mean = keep_f32(std::make_shared<ov::op::v1::ReduceMean>(x_f32(), axes, true));
    auto deviation = keep_f32(std::make_shared<ov::op::v1::Subtract>(x_f32(), mean));
    auto squared_deviation = keep_f32(std::make_shared<ov::op::v1::Multiply>(deviation, deviation));
    auto variance = keep_f32(std::make_shared<ov::op::v1::ReduceMean>(squared_deviation, axes, true));
    auto std_dev = keep_f32(std::make_shared<ov::op::v0::Sqrt>(variance));
    auto k = keep_f32(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.6448536f}));
    auto scaled_std_dev = keep_f32(std::make_shared<ov::op::v1::Multiply>(std_dev, k));
    auto threshold_mean = variant == Variant::SeparateMeans ? keep_f32(std::make_shared<ov::op::v1::ReduceMean>(x_f32(), axes, true)) : mean;
    auto threshold = keep_f32(std::make_shared<ov::op::v1::Add>(threshold_mean, scaled_std_dev));
    auto sparsified = keep_f32(std::make_shared<ov::op::v1::Subtract>(x_f32(), threshold));
    auto relu = keep_f32(std::make_shared<ov::op::v0::Relu>(sparsified));

    auto relu_f16 = std::make_shared<ov::op::v0::Convert>(relu, ov::element::f16);
    ov::disable_constant_folding(relu_f16);
    auto zero = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.f});
    auto mask = std::make_shared<ov::op::v1::Greater>(relu_f16, zero);
    return std::make_shared<ov::Model>(ov::OutputVector{mask}, ov::ParameterVector{x});
}

// ConvertPrecision rounds a floating-point model output to FP16 and back and loses its tensor name, which the test fixture
// checks, so the model ends with a comparison.
std::shared_ptr<ov::Model> add_mask_output(const std::shared_ptr<ov::Model>& model) {
    auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.f});
    auto mask = std::make_shared<ov::op::v1::Greater>(model->get_results()[0]->input_value(0), zero);
    return std::make_shared<ov::Model>(ov::OutputVector{mask}, model->get_parameters());
}

// The comparator only requires the marks of model_ref to be present in the model, so the exact set of marked nodes is
// asserted as well.
class DisableFP16CompForGaussianTopKTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<DisableFP16CompForGaussianTopK>();
    }

    void TearDown() override {
        TransformationTestsF::TearDown();

        size_t matched_count = 0;
        for (const auto& op : model->get_ops()) {
            const bool is_marked = marked_nodes.count(op->get_friendly_name()) > 0;
            matched_count += is_marked;
            ASSERT_EQ(ov::is_conversion_disabled(op, ov::element::f16), is_marked) << "Node: " << op->get_friendly_name();
        }
        ASSERT_EQ(matched_count, marked_nodes.size()) << "Some marked nodes are missing in the model";
    }

    std::unordered_set<std::string> marked_nodes;
};

using PositiveParams = std::tuple<Variant, bool>;  // variant, mean is the first input of the threshold

class DisableFP16CompForGaussianTopKPositiveTest : public testing::WithParamInterface<PositiveParams>, public DisableFP16CompForGaussianTopKTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PositiveParams>& obj) {
        const auto& [variant, mean_first] = obj.param;
        return std::string(variant == Variant::GaussianTopK ? "SharedMean" : "SeparateMeans") + (mean_first ? "_MeanFirst" : "_ScaledStdFirst");
    }
};

class DisableFP16CompForGaussianTopKNoOpTest : public testing::WithParamInterface<Variant>, public DisableFP16CompForGaussianTopKTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<Variant>& obj) {
        switch (obj.param) {
        case Variant::NonSquaredDeviation:
            return "NonSquaredDeviation";
        case Variant::ThresholdFromOtherMean:
            return "ThresholdFromOtherMean";
        case Variant::ThresholdMeanKeepDimsMismatch:
            return "ThresholdMeanKeepDimsMismatch";
        case Variant::MismatchedReduceAxes:
            return "MismatchedReduceAxes";
        case Variant::ThresholdMeanOfOtherInput:
            return "ThresholdMeanOfOtherInput";
        default:
            return "Unknown";
        }
    }
};

// ConvertPrecision is registered as in the GPU plugin: the mark set by the pass propagates up to mean and deviation.
class DisableFP16CompForGaussianTopKConvertPrecisionTest : public testing::WithParamInterface<Variant>, public TransformationTestsF {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<Variant>& obj) {
        return obj.param == Variant::GaussianTopK ? "SharedMean" : "SeparateMeans";
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<DisableFP16CompForGaussianTopK>();
        precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
        manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                          type_to_fuse_map{},
                                                          true,   // keep_precision_sensitive_in_fp32
                                                          false,  // convert_input_output_precision
                                                          true);  // store_original_precision_as_rt_attribute
    }

    void TearDown() override {
        TransformationTestsF::TearDown();

        std::unordered_set<std::string> fp32_nodes =
            {name_mean, name_deviation, name_squared_deviation, name_variance, name_std_dev, name_scaled_std_dev, name_threshold};
        if (GetParam() == Variant::SeparateMeans)
            fp32_nodes.insert(name_threshold_mean);

        size_t matched_count = 0;
        for (const auto& op : model->get_ops()) {
            if (fp32_nodes.count(op->get_friendly_name()) == 0)
                continue;
            ++matched_count;
            ASSERT_EQ(op->get_output_element_type(0), ov::element::f32) << "Node is converted to FP16: " << op->get_friendly_name();
        }
        ASSERT_EQ(matched_count, fp32_nodes.size()) << "Some FP32 nodes are missing in the model";
    }
};
}  // namespace

// mean and deviation are not marked by the pass: ConvertPrecision propagates the mark up to them
// (see DisableFP16CompForGaussianTopKConvertPrecisionTest).
TEST_P(DisableFP16CompForGaussianTopKPositiveTest, MarksOverflowProneNodes) {
    const auto& [variant, mean_first] = GetParam();
    model = create_model(variant, mean_first);
    model_ref = create_model(variant, mean_first, true);
    marked_nodes = {name_squared_deviation, name_variance, name_std_dev, name_scaled_std_dev, name_threshold};
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         DisableFP16CompForGaussianTopKPositiveTest,
                         testing::Combine(testing::Values(Variant::GaussianTopK, Variant::SeparateMeans), testing::Bool()),
                         DisableFP16CompForGaussianTopKPositiveTest::getTestCaseName);

TEST_P(DisableFP16CompForGaussianTopKConvertPrecisionTest, KeepsMeanAndDeviationInFP32) {
    model = add_mask_output(create_model(GetParam()));
    model_ref = create_mixed_precision_model(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         DisableFP16CompForGaussianTopKConvertPrecisionTest,
                         testing::Values(Variant::GaussianTopK, Variant::SeparateMeans),
                         DisableFP16CompForGaussianTopKConvertPrecisionTest::getTestCaseName);

TEST_P(DisableFP16CompForGaussianTopKNoOpTest, NothingIsMarked) {
    model = create_model(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         DisableFP16CompForGaussianTopKNoOpTest,
                         testing::Values(Variant::NonSquaredDeviation,
                                         Variant::ThresholdFromOtherMean,
                                         Variant::ThresholdMeanKeepDimsMismatch,
                                         Variant::MismatchedReduceAxes,
                                         Variant::ThresholdMeanOfOtherInput),
                         DisableFP16CompForGaussianTopKNoOpTest::getTestCaseName);
