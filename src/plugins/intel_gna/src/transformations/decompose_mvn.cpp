// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/decompose_mvn.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/cc/ngraph/itt.hpp>
#include <transformations/utils/utils.hpp>

#include "backend/gna_limitations.hpp"

using namespace ngraph;
using namespace ov::intel_gna::limitations;

namespace ov {
namespace intel_gna {
namespace pass {

struct MVNData {
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    size_t num_parts;
    float eps;
    ov::op::MVNEpsMode eps_mode;
    bool normalize_variance;
    element::Type element_type;
    std::string name;
};

template <class T>
static bool ValidateAxes(const std::shared_ptr<opset8::Constant> axes_const, const size_t& mvn_shape_size) {
    T axes_value;
    size_t axes_vector_size;

    std::vector<T> axes_const_vector = axes_const->cast_vector<T>();
    IE_ASSERT(!axes_const_vector.empty());
    axes_value = axes_const_vector[0];
    axes_vector_size = axes_const_vector.size();

    if (axes_vector_size != mvn_shape_size - 2) {
        return false;
    }

    // Verify supported first axes value
    if (axes_value != 2 && axes_value != static_cast<T>(2 - mvn_shape_size))
        return false;

    return true;
}

static bool GetVerifiedMVNData(const std::shared_ptr<opset8::MVN> mvn, MVNData& mvn_data) {
    const auto mvn_shape = mvn->get_output_shape(0);
    auto mvn_shape_size = mvn_shape.size();

    // Validate axes parameter
    auto axes_const = std::dynamic_pointer_cast<opset8::Constant>(mvn->input_value(1).get_node_shared_ptr());
    IE_ASSERT(axes_const);
    auto element_type = axes_const->get_element_type();

    if (!(element_type == element::Type_t::i64 ? ValidateAxes<int64_t>(axes_const, mvn_shape_size)
                                               : ValidateAxes<int32_t>(axes_const, mvn_shape_size)))
        return false;

    if (mvn_shape_size == 4) {
        mvn_data.N = mvn_shape[0];
        mvn_data.C = mvn_shape[1];
        mvn_data.H = mvn_shape[2];
        mvn_data.W = mvn_shape[3];
    } else if (mvn_shape_size == 3) {
        mvn_data.N = 1;
        mvn_data.C = mvn_shape[0];
        mvn_data.H = mvn_shape[1];
        mvn_data.W = mvn_shape[2];
    } else {
        THROW_GNA_EXCEPTION << "Unsupported MVN shape size: " << mvn_shape_size;
    }

    // Check if average must be split
    mvn_data.num_parts = 1;
    while (mvn_data.W / mvn_data.num_parts > Limitations::kConvFilterMaxSize) {
        mvn_data.num_parts *= 2;
    }

    // Abort if W is not divisible by power of 2
    if ((mvn_data.W / mvn_data.num_parts) * mvn_data.num_parts != mvn_data.W) {
        return false;
    }

    mvn_data.eps = mvn->get_eps();
    mvn_data.eps_mode = mvn->get_eps_mode();
    mvn_data.normalize_variance = mvn->get_normalize_variance();
    mvn_data.element_type = mvn->get_element_type();
    mvn_data.name = mvn->get_friendly_name();

    return true;
}

static std::shared_ptr<Node> NormalizeVariance(const std::shared_ptr<opset8::MVN> mvn,
                                               const MVNData& mvn_data,
                                               const std::shared_ptr<opset8::Add>& subtract_mean,
                                               const std::shared_ptr<opset8::Constant>& avg_broadcast_const) {
    // Prepare consts
    auto combined_C_H = mvn_data.C * mvn_data.H;

    std::vector<float> avg_weights(8 * mvn_data.W / mvn_data.num_parts, 1.0f / mvn_data.W);
    auto avg_weights_const =
        opset8::Constant::create(mvn_data.element_type, Shape{8, mvn_data.W / mvn_data.num_parts, 1, 1}, avg_weights);
    std::vector<float> eps_tensor(combined_C_H * mvn_data.W, mvn_data.eps);
    auto eps_tensor_const =
        opset8::Constant::create(mvn_data.element_type, Shape{1, combined_C_H * mvn_data.W}, eps_tensor);
    std::vector<float> minus_half(combined_C_H * mvn_data.W, -0.5f);
    auto minus_half_const =
        opset8::Constant::create(mvn_data.element_type, Shape{1, combined_C_H * mvn_data.W}, minus_half);

    // Calculate square of the difference between input and its mean
    auto squared_diff = std::make_shared<opset8::Multiply>(subtract_mean, subtract_mean);
    squared_diff->set_friendly_name(mvn_data.name + "_SqrDiff");

    // Calculate sum of the squares
    auto squared_diff_reshape = std::make_shared<opset8::Reshape>(
        squared_diff,
        opset8::Constant::create(
            element::i64,
            Shape{4},
            Shape{mvn_data.N, combined_C_H * mvn_data.num_parts, 1ull, mvn_data.W / mvn_data.num_parts}),
        false);
    auto transposed_input_3 =
        std::make_shared<opset8::Transpose>(squared_diff_reshape,
                                            opset8::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2}));
    auto transposed_avg_conv_3 = std::make_shared<opset8::Convolution>(transposed_input_3,
                                                                       avg_weights_const,
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1},
                                                                       ov::op::PadType::VALID);
    transposed_avg_conv_3->set_friendly_name(mvn_data.name + "_Avg3");
    auto avg_conv_3 =
        std::make_shared<opset8::Transpose>(transposed_avg_conv_3,
                                            opset8::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1}));
    auto reshape_avg_conv_3 = std::make_shared<opset8::Reshape>(
        avg_conv_3,
        opset8::Constant::create(element::i64, Shape{4}, Shape{mvn_data.N, 1ull, combined_C_H, 8 * mvn_data.num_parts}),
        false);
    auto transposed_input_4 =
        std::make_shared<opset8::Transpose>(reshape_avg_conv_3,
                                            opset8::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2}));
    auto transposed_avg_conv_4 = std::make_shared<opset8::Convolution>(transposed_input_4,
                                                                       avg_broadcast_const,
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1},
                                                                       ov::op::PadType::VALID);
    transposed_avg_conv_4->set_friendly_name(mvn_data.name + "_Avg4");
    auto avg_conv_4 =
        std::make_shared<opset8::Transpose>(transposed_avg_conv_4,
                                            opset8::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1}));
    auto reshape_avg_conv_4 = std::make_shared<opset8::Reshape>(
        avg_conv_4,
        opset8::Constant::create(element::i64, Shape{2}, Shape{1ull, combined_C_H * mvn_data.W}),
        false);
    std::shared_ptr<Node> inv_stdev;

    // Create normalization part of the graph
    // We ignore inside/outside epsilon position here and always use inside, to get better accuracy
    // even though the built-in MVN1 to MVN6 transformation enforces outside setting

    // Add epsilon inside the square root
    auto add_epsilon = std::make_shared<opset8::Add>(eps_tensor_const, reshape_avg_conv_4);

    // Calculate square root and inversion
    auto log_var_eps = std::make_shared<opset8::Log>(add_epsilon);
    log_var_eps->set_friendly_name(mvn_data.name + "_LogVarEps");
    auto log_inv_stdev = std::make_shared<opset8::Multiply>(log_var_eps, minus_half_const);
    log_inv_stdev->set_friendly_name(mvn_data.name + "_LogInvStdev");
    inv_stdev = std::make_shared<opset8::Exp>(log_inv_stdev);
    inv_stdev->set_friendly_name(mvn_data.name + "_InvStdev");
    copy_runtime_info(mvn, {add_epsilon, log_var_eps, log_inv_stdev, inv_stdev});

    auto normalized_output = std::make_shared<opset8::Multiply>(subtract_mean, inv_stdev);
    normalized_output->set_friendly_name(mvn_data.name + "_Output");

    copy_runtime_info(mvn,
                      {squared_diff,
                       squared_diff_reshape,
                       transposed_input_3,
                       transposed_avg_conv_3,
                       avg_conv_3,
                       reshape_avg_conv_3,
                       transposed_input_4,
                       transposed_avg_conv_4,
                       avg_conv_4,
                       reshape_avg_conv_4});

    return normalized_output;
}

static void Decompose(const std::shared_ptr<opset8::MVN> mvn, const MVNData& mvn_data) {
    // Prepare data
    auto combined_C_H = mvn_data.C * mvn_data.H;

    std::vector<float> neg_avg_weights(8 * mvn_data.W / mvn_data.num_parts, -1.0f / mvn_data.W);
    auto neg_avg_weights_const = opset8::Constant::create(mvn_data.element_type,
                                                          Shape{8, mvn_data.W / mvn_data.num_parts, 1, 1},
                                                          neg_avg_weights);

    std::vector<float> avg_broadcast(8 * mvn_data.W * mvn_data.num_parts, 0.0f);
    for (size_t i = 0; i < mvn_data.W * mvn_data.num_parts; i++) {
        avg_broadcast[i * 8] = 1.0f;
    }
    auto avg_broadcast_const =
        opset8::Constant::create(mvn_data.element_type, Shape{mvn_data.W, 8 * mvn_data.num_parts, 1, 1}, avg_broadcast);

    // Create average calculation part of the graph
    // We assume C = 1 case (combined channels)
    const auto input = mvn->input_value(0);
    auto reshape = std::make_shared<opset8::Reshape>(
        input,
        opset8::Constant::create(element::i64, Shape{4}, Shape{mvn_data.N, 1ull, combined_C_H, mvn_data.W}),
        false);
    auto input_4d = std::make_shared<opset8::Reshape>(
        reshape,
        opset8::Constant::create(
            element::i64,
            Shape{4},
            Shape{mvn_data.N, combined_C_H * mvn_data.num_parts, 1ull, mvn_data.W / mvn_data.num_parts}),
        false);
    auto input_2d = std::make_shared<opset8::Reshape>(
        reshape,
        opset8::Constant::create(element::i64, Shape{2}, Shape{1ull, combined_C_H * mvn_data.W}),
        false);
    auto transposed_input_1 =
        std::make_shared<opset8::Transpose>(input_4d, opset8::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2}));
    auto transposed_avg_conv_1 = std::make_shared<opset8::Convolution>(transposed_input_1,
                                                                       neg_avg_weights_const,
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1},
                                                                       ov::op::PadType::VALID);
    transposed_avg_conv_1->set_friendly_name(mvn_data.name + "_Avg1");
    auto avg_conv_1 =
        std::make_shared<opset8::Transpose>(transposed_avg_conv_1,
                                            opset8::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1}));
    auto reshape_avg_conv_1 = std::make_shared<opset8::Reshape>(
        avg_conv_1,
        opset8::Constant::create(element::i64, Shape{4}, Shape{mvn_data.N, 1ull, combined_C_H, 8 * mvn_data.num_parts}),
        false);
    auto transposed_input_2 =
        std::make_shared<opset8::Transpose>(reshape_avg_conv_1,
                                            opset8::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2}));
    auto transposed_avg_conv_2 = std::make_shared<opset8::Convolution>(transposed_input_2,
                                                                       avg_broadcast_const,
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1},
                                                                       ov::op::PadType::VALID);
    transposed_avg_conv_2->set_friendly_name(mvn_data.name + "_Avg2");
    auto avg_conv_2 =
        std::make_shared<opset8::Transpose>(transposed_avg_conv_2,
                                            opset8::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1}));
    auto avg_conv_2_2d = std::make_shared<opset8::Reshape>(
        avg_conv_2,
        opset8::Constant::create(element::i64, Shape{2}, Shape{1ull, combined_C_H * mvn_data.W}),
        false);
    auto subtract_mean = std::make_shared<opset8::Add>(input_2d, avg_conv_2_2d);
    subtract_mean->set_friendly_name(mvn_data.name + "_SubMean");

    std::shared_ptr<Node> mvn_output, pre_output = subtract_mean;

    // Normalize variance if required
    if (mvn_data.normalize_variance) {
        pre_output = NormalizeVariance(mvn, mvn_data, subtract_mean, avg_broadcast_const);
    }

    // Reshape (combined channels) back to get the final output
    if (mvn->get_output_shape(0).size() == 3) {
        mvn_output = std::make_shared<opset8::Reshape>(
            pre_output,
            opset8::Constant::create(element::i64, Shape{3}, {mvn_data.C, mvn_data.H, mvn_data.W}),
            false);
    } else {
        mvn_output = std::make_shared<opset8::Reshape>(
            pre_output,
            opset8::Constant::create(element::i64, Shape{4}, {mvn_data.N, mvn_data.C, mvn_data.H, mvn_data.W}),
            false);
    }

    copy_runtime_info(mvn,
                      {reshape,
                       input_4d,
                       input_2d,
                       transposed_input_1,
                       transposed_avg_conv_1,
                       avg_conv_1,
                       reshape_avg_conv_1,
                       transposed_input_2,
                       transposed_avg_conv_2,
                       avg_conv_2,
                       avg_conv_2_2d,
                       subtract_mean,
                       mvn_output});

    // We need retain the MVN layer name, so its output can be used as a network result
    replace_node(mvn, mvn_output);
    mvn_output->set_friendly_name(mvn_data.name);
}

static bool Convert(std::shared_ptr<Node> mvn_node) {
    const auto mvn = std::dynamic_pointer_cast<opset8::MVN>(mvn_node);
    MVNData mvn_data = {};

    if (!GetVerifiedMVNData(mvn, mvn_data))
        return false;

    Decompose(mvn, mvn_data);

    return true;
}

static std::function<bool(Output<Node>)> verify_rank_batch() {
    return [=](Output<Node> output) -> bool {
        // Only rank 3 and 4 and batch 1 are supported for now
        auto rank = output.get_partial_shape().rank();
        if (rank != 3 && rank != 4)
            return false;

        auto batch = (rank == 3 ? 1 : output.get_partial_shape()[0]);
        if (batch != 1)
            return false;

        return true;
    };
}

DecomposeMVN::DecomposeMVN() {
    MATCHER_SCOPE(DecomposeMVN);

    auto axes = pattern::wrap_type<opset8::Constant>();
    auto mvn = pattern::wrap_type<opset8::MVN>({pattern::any_input(), axes}, verify_rank_batch());

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return Convert(pattern_map.at(mvn).get_node_shared_ptr());
    };

    auto m = std::make_shared<pattern::Matcher>(mvn, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
