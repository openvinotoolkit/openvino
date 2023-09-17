// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <tuple>

#include "backend/gna_limitations.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/decompose_mvn.hpp"
#include "transformations/op_conversions/convert_mvn1_to_mvn6.hpp"

using namespace ov::intel_gna::limitations;

namespace decomposeMVN {

typedef std::tuple<ngraph::Shape,                // Input shape
                   bool,                         // Normalize variance
                   float,                        // Epsilon
                   ngraph::op::MVNEpsMode,       // Epsilon mode
                   InferenceEngine::SizeVector,  // Axes tensor
                   bool,                         // Across channels
                   bool                          // MVN version, true = v6, false = v1
                   >
    decomposeMVNParams;

struct MVNParams {
    size_t N;
    size_t C;
    size_t H;
    size_t W = 0;
    size_t num_parts;
    float eps;
    ngraph::op::MVNEpsMode eps_mode;
    bool normalize_variance;
};

static std::shared_ptr<ngraph::Node> NormalizeVariance(
    const MVNParams& mvn_data,
    const std::shared_ptr<ngraph::opset8::Add>& subtract_mean,
    const std::shared_ptr<ngraph::opset8::Constant>& avg_broadcast_const) {
    // Prepare consts
    auto combined_C_H = mvn_data.C * mvn_data.H;

    std::vector<float> avg_weights(8 * mvn_data.W / mvn_data.num_parts, 1.0f / mvn_data.W);
    auto avg_weights_const = ngraph::opset8::Constant::create(ngraph::element::f32,
                                                              ngraph::Shape{8, mvn_data.W / mvn_data.num_parts, 1, 1},
                                                              avg_weights);
    std::vector<float> eps_tensor(combined_C_H * mvn_data.W, mvn_data.eps);
    auto eps_tensor_const =
        ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1, combined_C_H * mvn_data.W}, eps_tensor);
    std::vector<float> minus_half(combined_C_H * mvn_data.W, -0.5f);
    auto minus_half_const =
        ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1, combined_C_H * mvn_data.W}, minus_half);

    // Calculate square of the difference between input and its mean
    auto squared_diff = std::make_shared<ngraph::opset8::Multiply>(subtract_mean, subtract_mean);
    squared_diff->set_friendly_name("MvnSqrDiff");

    // Calculate sum of the squares
    auto squared_diff_reshape = std::make_shared<ngraph::opset8::Reshape>(
        squared_diff,
        ngraph::opset8::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{4},
            ngraph::Shape{mvn_data.N, combined_C_H * mvn_data.num_parts, 1ull, mvn_data.W / mvn_data.num_parts}),
        false);
    auto transposed_input_3 = std::make_shared<ngraph::opset8::Transpose>(
        squared_diff_reshape,
        ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2}));
    auto transposed_avg_conv_3 = std::make_shared<ngraph::opset8::Convolution>(transposed_input_3,
                                                                               avg_weights_const,
                                                                               ngraph::Strides{1, 1},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::Strides{1, 1},
                                                                               ngraph::op::PadType::VALID);
    transposed_avg_conv_3->set_friendly_name("MvnAvg3");
    auto avg_conv_3 = std::make_shared<ngraph::opset8::Transpose>(
        transposed_avg_conv_3,
        ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1}));
    auto reshape_avg_conv_3 = std::make_shared<ngraph::opset8::Reshape>(
        avg_conv_3,
        ngraph::opset8::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{4},
                                         ngraph::Shape{mvn_data.N, 1ull, combined_C_H, 8 * mvn_data.num_parts}),
        false);
    auto transposed_input_4 = std::make_shared<ngraph::opset8::Transpose>(
        reshape_avg_conv_3,
        ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2}));
    auto transposed_avg_conv_4 = std::make_shared<ngraph::opset8::Convolution>(transposed_input_4,
                                                                               avg_broadcast_const,
                                                                               ngraph::Strides{1, 1},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::Strides{1, 1},
                                                                               ngraph::op::PadType::VALID);
    transposed_avg_conv_4->set_friendly_name("MvnAvg4");
    auto avg_conv_4 = std::make_shared<ngraph::opset8::Transpose>(
        transposed_avg_conv_4,
        ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1}));
    auto reshape_avg_conv_4 = std::make_shared<ngraph::opset8::Reshape>(
        avg_conv_4,
        ngraph::opset8::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{2},
                                         ngraph::Shape{1ull, combined_C_H * mvn_data.W}),
        false);
    std::shared_ptr<ngraph::Node> inv_stdev;

    // Create normalization part of the graph
    // We ignore inside/outside epsilon position here and always use inside, to get better accuracy
    // even though the built-in MVN1 to MVN6 transformation enforces outside setting

    // Add epsilon inside the square root
    auto add_epsilon = std::make_shared<ngraph::opset8::Add>(eps_tensor_const, reshape_avg_conv_4);

    // Calculate square root and inversion
    auto log_var_eps = std::make_shared<ngraph::opset8::Log>(add_epsilon);
    log_var_eps->set_friendly_name("MvnLogVarEps");
    auto log_inv_stdev = std::make_shared<ngraph::opset8::Multiply>(log_var_eps, minus_half_const);
    log_inv_stdev->set_friendly_name("MvnLogInvStdev");
    inv_stdev = std::make_shared<ngraph::opset8::Exp>(log_inv_stdev);
    inv_stdev->set_friendly_name("MvnInvStdev");

    auto normalized_output = std::make_shared<ngraph::opset8::Multiply>(subtract_mean, inv_stdev);
    normalized_output->set_friendly_name("MvnOutput");

    return normalized_output;
}

static std::shared_ptr<ngraph::opset8::Result> Decompose(const std::shared_ptr<ngraph::Node> input_node,
                                                         const MVNParams& mvn_data) {
    // Prepare data
    auto combined_C_H = mvn_data.C * mvn_data.H;

    std::vector<float> neg_avg_weights(8 * mvn_data.W / mvn_data.num_parts, -1.0f / mvn_data.W);
    auto neg_avg_weights_const =
        ngraph::opset8::Constant::create(ngraph::element::f32,
                                         ngraph::Shape{8, mvn_data.W / mvn_data.num_parts, 1, 1},
                                         neg_avg_weights);

    std::vector<float> avg_broadcast(8 * mvn_data.W * mvn_data.num_parts, 0.0f);
    for (size_t i = 0; i < mvn_data.W * mvn_data.num_parts; i++) {
        avg_broadcast[i * 8] = 1.0f;
    }
    auto avg_broadcast_const = ngraph::opset8::Constant::create(ngraph::element::f32,
                                                                ngraph::Shape{mvn_data.W, 8 * mvn_data.num_parts, 1, 1},
                                                                avg_broadcast);

    // Create average calculation part of the graph
    // We assume C = 1 case (combined channels)
    auto reshape = std::make_shared<ngraph::opset8::Reshape>(
        input_node,
        ngraph::opset8::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{4},
                                         ngraph::Shape{mvn_data.N, 1ull, combined_C_H, mvn_data.W}),
        false);
    auto input_4d = std::make_shared<ngraph::opset8::Reshape>(
        reshape,
        ngraph::opset8::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{4},
            ngraph::Shape{mvn_data.N, combined_C_H * mvn_data.num_parts, 1ull, mvn_data.W / mvn_data.num_parts}),
        false);
    auto input_2d = std::make_shared<ngraph::opset8::Reshape>(
        reshape,
        ngraph::opset8::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{2},
                                         ngraph::Shape{1ull, combined_C_H * mvn_data.W}),
        false);
    auto transposed_input_1 = std::make_shared<ngraph::opset8::Transpose>(
        input_4d,
        ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2}));
    auto transposed_avg_conv_1 = std::make_shared<ngraph::opset8::Convolution>(transposed_input_1,
                                                                               neg_avg_weights_const,
                                                                               ngraph::Strides{1, 1},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::Strides{1, 1},
                                                                               ngraph::op::PadType::VALID);
    transposed_avg_conv_1->set_friendly_name("MvnAvg1");
    auto avg_conv_1 = std::make_shared<ngraph::opset8::Transpose>(
        transposed_avg_conv_1,
        ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1}));
    auto reshape_avg_conv_1 = std::make_shared<ngraph::opset8::Reshape>(
        avg_conv_1,
        ngraph::opset8::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{4},
                                         ngraph::Shape{mvn_data.N, 1ull, combined_C_H, 8 * mvn_data.num_parts}),
        false);
    auto transposed_input_2 = std::make_shared<ngraph::opset8::Transpose>(
        reshape_avg_conv_1,
        ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2}));
    auto transposed_avg_conv_2 = std::make_shared<ngraph::opset8::Convolution>(transposed_input_2,
                                                                               avg_broadcast_const,
                                                                               ngraph::Strides{1, 1},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::Strides{1, 1},
                                                                               ngraph::op::PadType::VALID);
    transposed_avg_conv_2->set_friendly_name("MvnAvg2");
    auto avg_conv_2 = std::make_shared<ngraph::opset8::Transpose>(
        transposed_avg_conv_2,
        ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1}));
    auto avg_conv_2_2d = std::make_shared<ngraph::opset8::Reshape>(
        avg_conv_2,
        ngraph::opset8::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{2},
                                         ngraph::Shape{1ull, combined_C_H * mvn_data.W}),
        false);
    auto subtract_mean = std::make_shared<ngraph::opset8::Add>(input_2d, avg_conv_2_2d);
    subtract_mean->set_friendly_name("MvnSubMean");

    std::shared_ptr<ngraph::Node> mvn_output, pre_output = subtract_mean;

    // Normalize variance if required
    if (mvn_data.normalize_variance) {
        pre_output = NormalizeVariance(mvn_data, subtract_mean, avg_broadcast_const);
    }

    // Reshape (combined channels) back to get the final output
    if (input_node->get_output_shape(0).size() == 3) {
        mvn_output = std::make_shared<ngraph::opset8::Reshape>(
            pre_output,
            ngraph::opset8::Constant::create(ngraph::element::i64,
                                             ngraph::Shape{3},
                                             {mvn_data.C, mvn_data.H, mvn_data.W}),
            false);
    } else {
        mvn_output = std::make_shared<ngraph::opset8::Reshape>(
            pre_output,
            ngraph::opset8::Constant::create(ngraph::element::i64,
                                             ngraph::Shape{4},
                                             {mvn_data.N, mvn_data.C, mvn_data.H, mvn_data.W}),
            false);
    }

    return std::make_shared<ngraph::opset8::Result>(mvn_output);
}

std::shared_ptr<ngraph::Function> getReferenceFunction(const ngraph::Shape& input_shape,
                                                       const bool& normalize_variance,
                                                       const float& eps,
                                                       const ngraph::op::MVNEpsMode& eps_mode,
                                                       const InferenceEngine::SizeVector& axes) {
    MVNParams mvn_data;
    auto mvn_shape_size = input_shape.size();

    if (mvn_shape_size == 4) {
        mvn_data.N = input_shape[0];
        mvn_data.C = input_shape[1];
        mvn_data.H = input_shape[2];
        mvn_data.W = input_shape[3];
    } else if (mvn_shape_size == 3) {
        mvn_data.N = 1;
        mvn_data.C = input_shape[0];
        mvn_data.H = input_shape[1];
        mvn_data.W = input_shape[2];
    }

    mvn_data.eps = eps;
    mvn_data.eps_mode = eps_mode;
    mvn_data.normalize_variance = normalize_variance;
    mvn_data.num_parts = 1;

    while (mvn_data.W / mvn_data.num_parts > Limitations::kConvFilterMaxSize) {
        mvn_data.num_parts *= 2;
    }

    // Create decomposed reference function
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
    std::shared_ptr<ngraph::opset8::Result> result = Decompose(input_params, mvn_data);

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

std::shared_ptr<ngraph::Function> getInitialFunction(const ngraph::Shape& input_shape,
                                                     const bool& normalize_variance,
                                                     const float& eps,
                                                     const ngraph::op::MVNEpsMode& eps_mode,
                                                     const InferenceEngine::SizeVector& axes,
                                                     const bool& across_channels,
                                                     const bool& mvn_version_6) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
    std::shared_ptr<ngraph::Node> mvn;

    if (mvn_version_6) {
        const auto axesConst =
            std::make_shared<ngraph::opset8::Constant>(ngraph::element::i64, ngraph::Shape{axes.size()}, axes);
        mvn = std::make_shared<ngraph::opset8::MVN>(input_params, axesConst, normalize_variance, eps, eps_mode);
    } else {
        mvn = std::make_shared<ngraph::opset2::MVN>(input_params, across_channels, normalize_variance, eps);
    }

    auto result = std::make_shared<ngraph::opset8::Result>(mvn);

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

}  // namespace decomposeMVN

// ---------------------------------------------------------------------------------------------------------------------

namespace {

void execute_test(std::shared_ptr<ngraph::Function> function, std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::ConvertMVN1ToMVN6>();
    manager.register_pass<ov::intel_gna::pass::DecomposeMVN>();
    manager.run_passes(function);
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

}  // namespace

TEST(TransformationTests, DecomposeMVNTest) {
    for (auto mvn_version_6 : {true, false}) {
        for (auto normalize_variance : {true, false}) {
            execute_test(decomposeMVN::getInitialFunction(ngraph::Shape{1, 1, 5, 300},
                                                          normalize_variance,
                                                          1.0e-09f,
                                                          ngraph::op::MVNEpsMode::INSIDE_SQRT,
                                                          InferenceEngine::SizeVector{2, 1},
                                                          false,
                                                          mvn_version_6),
                         decomposeMVN::getReferenceFunction(ngraph::Shape{1, 1, 5, 300},
                                                            normalize_variance,
                                                            1.0e-09f,
                                                            ngraph::op::MVNEpsMode::INSIDE_SQRT,
                                                            InferenceEngine::SizeVector{2, 1}));
            execute_test(decomposeMVN::getInitialFunction(ngraph::Shape{1, 6, 256},
                                                          normalize_variance,
                                                          1.0e-09f,
                                                          ngraph::op::MVNEpsMode::INSIDE_SQRT,
                                                          InferenceEngine::SizeVector{2},
                                                          false,
                                                          mvn_version_6),
                         decomposeMVN::getReferenceFunction(ngraph::Shape{1, 6, 256},
                                                            normalize_variance,
                                                            1.0e-09f,
                                                            ngraph::op::MVNEpsMode::INSIDE_SQRT,
                                                            InferenceEngine::SizeVector{2}));
        }
    }
}
