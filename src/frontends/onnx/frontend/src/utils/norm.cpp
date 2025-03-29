// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/norm.hpp"

#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sqrt.hpp"

namespace ov {
namespace op {
namespace util {
namespace {
/// \brief      Specifies method of bias application to avoid numerical problems
enum class BiasMode {
    // Add bias to intermediate result
    ADD,
    // Calculate max of intermediate result and bias
    MAX
};

std::shared_ptr<ov::Node> lp_norm(const Output<ov::Node>& value,
                                  size_t p_norm,
                                  const Output<ov::Node>& reduction_axes,
                                  float bias,
                                  bool keep_dims) {
    // In general "entrywise" lp-norm for matrix `A` is defined as following double
    // sum:
    // ||A||_p = ||vec(A)||_p = [sum_{i=1}^m sum_{j=1}^n abs(a_{i,j})^p]^{1/p}
    std::shared_ptr<ov::Node> abs_values{std::make_shared<ov::op::v0::Abs>(value)};
    std::shared_ptr<ov::Node> p_node = ov::op::v0::Constant::create(value.get_element_type(), Shape{}, {p_norm});

    // Get inner part of equation: abs_values^p_node, then sum over reduction_axes.
    std::shared_ptr<ov::Node> values{std::make_shared<ov::op::v1::Power>(abs_values, p_node)};
    values = std::make_shared<ov::op::v1::ReduceSum>(values, reduction_axes, keep_dims);

    std::shared_ptr<ov::Node> bias_node{ov::op::v0::Constant::create(values->get_element_type(), Shape{}, {bias})};

    values = std::make_shared<ov::op::v1::Add>(values, bias_node);

    // Get outer part of equation: raise values to 1/p_norm exponent.
    std::shared_ptr<ov::Node> inv_p_node =
        ov::op::v0::Constant::create(values->get_element_type(), Shape{}, {1.f / p_norm});

    return std::make_shared<ov::op::v1::Power>(values, inv_p_node);
}

/// \brief      Calculates L-0 norm of input tensor.
///
/// \note       The L-0 norm represents the cardinality of elements different
///             from zero. This actually is not a "true" norm.
///
/// \param[in]  value           The input tensor.
/// \param[in]  reduction_axes  The axes along which we calculate norm.
/// \param[in]  keep_dims       The flag indicates if axes will be removed or kept.
///
/// \return     L-0 norm of value. The output sub-graph is composed of v1 ops.
///
std::shared_ptr<ov::Node> l0_norm(const Output<ov::Node>& value,
                                  const Output<ov::Node>& reduction_axes,
                                  bool keep_dims) {
    // L0 norm returns number of elements different from zero.
    const std::shared_ptr<ov::Node> zero_node{ov::op::v0::Constant::create(value.get_element_type(), Shape{}, {0.f})};

    // Convert bool values to input node data type.
    const std::shared_ptr<ov::Node> non_zero_values =
        std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v1::NotEqual>(value, zero_node),
                                              value.get_element_type());

    return std::make_shared<ov::op::v1::ReduceSum>(non_zero_values, reduction_axes, keep_dims);
}

/// \brief      Calculates L-1 norm of a value.
///
/// \note       The L-1 norm represents the sum of absolute values.
///
/// \param[in]  value           The input tensor.
/// \param[in]  reduction_axes  The axes along which we calculate norm.
/// \param[in]  bias            The bias added to the calculated sum.
/// \param[in]  keep_dims       The flag indicates if axes will be removed or kept.
///
/// \return     L-1 norm of value. The output sub-graph is composed of v1 ops.
///
std::shared_ptr<ov::Node> l1_norm(const Output<ov::Node>& value,
                                  const Output<ov::Node>& reduction_axes,
                                  float bias,
                                  bool keep_dims) {
    const std::shared_ptr<ov::Node> values{
        std::make_shared<ov::op::v1::ReduceSum>(std::make_shared<ov::op::v0::Abs>(value), reduction_axes, keep_dims)};

    const std::shared_ptr<ov::Node> bias_node{
        ov::op::v0::Constant::create(values->get_element_type(), Shape{}, {bias})};

    return std::make_shared<ov::op::v1::Add>(values, bias_node);
}

/// \brief      Calculates L-2 norm of input tensor.
///
/// \note       The L-2 norm represents the square root of sum of squares of each
///             individual element.
///
/// \param[in]  value           The input tensor.
/// \param[in]  reduction_axes  The axes along which we calculate norm.
/// \param[in]  bias            The bias combined with calculated sum.
/// \param[in]  bias_mode       The method of bias application.
/// \param[in]  keep_dims       The flag indicates if axes will be removed or kept.
///
/// \return     L-2 norm of value. The output sub-graph is composed of v1 ops.
///
std::shared_ptr<ov::Node> l2_norm(const Output<ov::Node>& value,
                                  const Output<ov::Node>& reduction_axes,
                                  float bias,
                                  BiasMode bias_mode,
                                  bool keep_dims) {
    std::shared_ptr<ov::Node> pow = std::make_shared<ov::op::v1::Power>(
        value,
        std::make_shared<ov::op::v0::Constant>(value.get_element_type(), Shape{}, 2));
    std::shared_ptr<ov::Node> values{std::make_shared<ov::op::v1::ReduceSum>(pow, reduction_axes, keep_dims)};

    std::shared_ptr<ov::Node> bias_node{ov::op::v0::Constant::create(values->get_element_type(), Shape{}, {bias})};
    switch (bias_mode) {
    case BiasMode::MAX: {
        return std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Maximum>(values, bias_node));
    }
    case BiasMode::ADD:
    default:
        return std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(values, bias_node));
    }
}
}  // namespace

std::shared_ptr<ov::Node> lp_norm(const Output<ov::Node>& value,
                                  const Output<ov::Node>& reduction_axes,
                                  size_t p_norm,
                                  float bias,
                                  bool keep_dims) {
    // The number of non-zero elements
    if (p_norm == 0) {
        return l0_norm(value, reduction_axes, keep_dims);
    }
    //  sum of absolute values.
    else if (p_norm == 1) {
        return l1_norm(value, reduction_axes, bias, keep_dims);
    }
    // sqrt of sum of squares - Euclidean norm
    else if (p_norm == 2) {
        return l2_norm(value, reduction_axes, bias, BiasMode::ADD, keep_dims);
    }
    // generic case
    else {
        return lp_norm(value, p_norm, reduction_axes, bias, keep_dims);
    }
}
}  // namespace util
}  // namespace op
}  // namespace ov
