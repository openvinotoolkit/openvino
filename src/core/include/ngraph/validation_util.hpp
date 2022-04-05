// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/variable_context.hpp"
#include "openvino/core/validation_util.hpp"

namespace ngraph {
using ov::evaluate_as_partial_shape;
using ov::get_constant_from_source;
using ov::infer_auto_padding;
using ov::infer_convolution_forward;
using ov::normalize_axes;
using ov::normalize_axis;

NGRAPH_API
Strides conv_default_strides(const Node* node, const PartialShape& data_batch_shape, const PartialShape& filters_shape);

NGRAPH_API
CoordinateDiff conv_default_padding(const Node* node,
                                    const PartialShape& data_batch_shape,
                                    const PartialShape& filters_shape);

NGRAPH_API
PartialShape infer_windowed_reduction_output_shape(const Node* node,
                                                   const PartialShape& data_shape,
                                                   const Strides& data_dilation,
                                                   const CoordinateDiff& data_padding_below,
                                                   const CoordinateDiff& data_padding_above,
                                                   const PartialShape& window_shape,
                                                   const Strides& window_strides,
                                                   const Strides& window_dilation,
                                                   bool is_window_all_in_padding_allowed,
                                                   bool ceil_mode = false);

void validate_conv_params_spatial_dimensions(const Node* node,
                                             const size_t num_spatial_dims,
                                             const op::PadType auto_pad,
                                             Strides& strides,
                                             Strides& dilations,
                                             CoordinateDiff& pads_begin,
                                             CoordinateDiff& pads_end);

/// \brief      Validates input shape ranks and infers convolution forward output shape.
///
/// \param[in] node              Node with convolution operation.
/// \param[in] data_batch_pshape Partial shape of data batch input.
/// \param[in] filters_pshape    Partial shape of filters input.
/// \param[in] auto_pad          Type of padding.
/// \param     strides           Strides.
/// \param     dilations         Dilations.
/// \param     pads_begin        Pads begin.
/// \param     pads_end          Pads end.
///
/// \return Partial shape of the output.
PartialShape validate_and_infer_convolution_forward_output_shape(const Node* node,
                                                                 const Rank& result_ps_rank,
                                                                 const PartialShape& data_batch_pshape,
                                                                 const PartialShape& filters_pshape,
                                                                 const op::PadType auto_pad,
                                                                 Strides& strides,
                                                                 Strides& dilations,
                                                                 CoordinateDiff& pads_begin,
                                                                 CoordinateDiff& pads_end);

NGRAPH_API
PartialShape infer_batched_pooling_forward(const Node* node,
                                           const PartialShape& data_batch_shape,
                                           const CoordinateDiff& data_padding_below,
                                           const CoordinateDiff& data_padding_above,
                                           const PartialShape& window_shape,
                                           const Strides& window_strides,
                                           bool is_window_all_in_padding_allowed,
                                           bool ceil_mode = false,
                                           const Strides& window_dilation = Strides{});

NGRAPH_API
std::tuple<element::Type, PartialShape, PartialShape> infer_batch_norm_forward(const Node* node,
                                                                               element::Type input_element_type,
                                                                               element::Type gamma_element_type,
                                                                               element::Type beta_element_type,
                                                                               element::Type mean_element_type,
                                                                               element::Type variance_element_type,
                                                                               const PartialShape& input_shape,
                                                                               const PartialShape& gamma_shape,
                                                                               const PartialShape& beta_shape,
                                                                               const PartialShape& mean_shape,
                                                                               const PartialShape& variance_shape);

NGRAPH_API
std::tuple<element::Type, PartialShape, PartialShape> infer_batch_norm_forward(const Node* node,
                                                                               element::Type input_element_type,
                                                                               element::Type gamma_element_type,
                                                                               element::Type beta_element_type,
                                                                               const PartialShape& input_shape,
                                                                               const PartialShape& gamma_shape,
                                                                               const PartialShape& beta_shape);

/// \brief Apply auto padding to padding_above and padding_below inputs
///        if all needed informations are known.
///
/// \param image_shape       The shape of input image.
/// \param filter_shape      The shape of filter input.
/// \param filter_strides    The strides of applied padding.
/// \param filter_dilations  The dilations of applied padding.
/// \param pad_type          The type of padding. Auto padding is applied only
///                          for SAME_UPPER and SAME_LOWER mode.
/// \param padding_above     The beginning of padding shape.
/// \param end               The beginning of padding shape.
///
/// \return true if auto padding was applied successfully (all needed informations such as
///         spatial dims are known), false otherwise.
NGRAPH_API
bool try_apply_auto_padding(const PartialShape& image_shape,
                            const Shape& filter_shape,
                            const Strides& filter_strides,
                            const Strides& filter_dilations,
                            const op::PadType pad_type,
                            CoordinateDiff& padding_above,
                            CoordinateDiff& padding_below);

NGRAPH_API
PartialShape infer_slice_shape(const Node* node,
                               const PartialShape& input_shape,
                               const std::vector<int64_t>& begin,
                               const std::vector<int64_t>& end,
                               const std::vector<int64_t>& strides,
                               const AxisSet& begin_mask,
                               const AxisSet& end_mask,
                               const AxisSet& new_axis_mask,
                               const AxisSet& shrink_axis_mask,
                               const AxisSet& ellipsis_mask);

/// \brief Try to compute the maximum value of value
/// \return (true, max_value) if can be determined, or (false, numeric_limits<uint64_t>::max())
/// if not.
/// \deprecated Use evaluate_upper_bound instead
NGRAPH_DEPRECATED("Use evaluate_upper_bound: it would return HostTensorPtr to the value instead of a pair")
NGRAPH_API std::pair<bool, uint64_t> maximum_value(const Output<Node>& value);

/// \brief Evaluates outputs, treating values in value_map as already computed. value_map is
/// updated.
/// \param value_map Key is RawNodeOutput in graph, value is the computed value. Updated by the
/// function.
/// \param output_tensor_map Tensors to use for particular outputs
/// \param outputs Root set of values to try to compute
/// \param evaluation_context Storage of additional settings and attributes that can be used
/// when evaluating the function. This additional information can be shared across nodes.
NGRAPH_API void evaluate_nodes(std::map<RawNodeOutput, HostTensorPtr>& value_map,
                               std::map<RawNodeOutput, HostTensorPtr>& output_tensor_map,
                               const OutputVector& outputs,
                               const EvaluationContext& evaluation_context = EvaluationContext());

/// \brief Evaluates lower value estimation of the output tensor. Traverses graph up to deduce
/// estimation through it.
/// \param Node output pointing to the tensor for estimation.
/// \return HostTensorPtr to estimated value if can be determined, or nullptr.
NGRAPH_API HostTensorPtr evaluate_lower_bound(const Output<Node>& output);

/// \brief Evaluates lower value estimation of the output tensor. Traverses graph up to deduce
/// estimation through it.
/// \param output Tensor to be estimated.
/// \return HostTensorPtr to estimated value if can be determined, or nullptr.
NGRAPH_API HostTensorPtr evaluate_upper_bound(const Output<Node>& output);

/// \brief Evaluates lower and upper value estimations of the output tensor. Traverses graph up
/// to deduce estimation through it.
/// \param output Node output pointing to the tensor for estimation.
/// \return pair with HostTensorPtrs for lower and upper value estimation. Each object in pair
/// could be HostTensorPtr to estimated value if particular bound can be determined, or nullptr.
NGRAPH_API std::pair<HostTensorPtr, HostTensorPtr> evaluate_both_bounds(const Output<Node>& output);

/// \brief Estimates upper bound for node output tensors using only upper bounds of the nodes
/// inputs.
/// \param node Operation to be performed
/// \param output_values Vector of HostTensorPtrs representing resulting upper value estimations
/// \return boolean status if value evaluation was successful.
NGRAPH_API bool default_upper_bound_evaluator(const Node* node, const HostTensorVector& output_values);
/// \brief Estimates lower bound for node output tensors using only lower bounds of the nodes
/// inputs.
/// \param node Operation to be performed
/// \param output_values Vector of HostTensorPtrs representing resulting lower value estimations
/// \return boolean status if value evaluation was successful.
NGRAPH_API bool default_lower_bound_evaluator(const Node* node, const HostTensorVector& output_values);
/// \brief Estimates both bounds for node output tensors using both bounds of inputs. Works for
/// operations with two inputs (in_1 and in_2). Brute forces all the pairs of bounds for inputs
/// and evaluates all of them: {in_1_lower, in_2 lower}, {in_1_lower, in_2 upper}, {in_1_upper,
/// in_2_lower}, {in_1_upper, in_2_upper}. Lower and upper values are selected from all the
/// outputs calculated using input pairs.
/// \param node Operation to be performed
/// \param output_values Vector of HostTensorPtrs representing resulting lower value estimations
/// \return boolean status if value evaluation was successful.
NGRAPH_API bool interval_bound_evaluator(const Node* node,
                                         const HostTensorVector& lower_output_values,
                                         const HostTensorVector& upper_output_values);

/// \brief Checks if all the elements of the bound HostTensor are positive
NGRAPH_API bool host_tensor_is_positive(const HostTensorPtr& bound);

/// \brief Checks if lower and upper bounds of the corresponding tensor are set (not nullptr)
/// and pointers are the same. It doesn't check if lower and upper values are the same relying
/// only on pointers comparison.
NGRAPH_API bool has_and_set_equal_bounds(const Output<Node>& source);

/// \brief Returns a Constant storing scalar value equal to std::numeric_limits<t>::max()
NGRAPH_API std::shared_ptr<op::Constant> get_constant_max_of_type(element::Type_t t);

/// \brief Returns a Constant storing scalar value equal to std::numeric_limits<t>::min()
NGRAPH_API std::shared_ptr<op::Constant> get_constant_min_of_type(element::Type_t t);

/// \brief Returns a Constant storing scalar value equal to std::numeric_limits<t>::lowest()
NGRAPH_API std::shared_ptr<op::Constant> get_constant_lowest_of_type(element::Type_t t);

/// \brief Checks if size of HostTensorVector is the same as passed size attribute. Then checks
/// that all the HostTensorPtrs are not equal to nullptr
NGRAPH_API bool validate_host_tensor_vector(const HostTensorVector& v, const size_t& size);

NGRAPH_API bool could_propagate(const Output<Node>& output, std::vector<Node*>& order);

namespace opset1 {
///
/// \brief      Calculates padding values for ConvolutionBackpropData operator.
///
/// \param[in]  input_data_shape  The input data shape.
/// \param[in]  filters_shape     The filters shape.
/// \param[in]  output_shape      The output shape defined only for spatial dimentions.
/// \param[in]  strides           The strides values.
/// \param[in]  dilations         The dilations values.
/// \param[in]  auto_pad_type     The automatic padding mode.
/// \param[in]  output_padding    The output padding values.
/// \param      pads_begin        The placeholder for paddings at the beginning of axis.
/// \param      pads_end          The placeholder for paddings at the end of axis.
///
NGRAPH_API
void infer_conv_backprop_auto_padding(const Shape& input_data_shape,
                                      const Shape& filters_shape,
                                      const Shape& output_shape,
                                      const Strides& strides,
                                      const Strides& dilations,
                                      const op::PadType auto_pad_type,
                                      const CoordinateDiff& output_padding,
                                      CoordinateDiff& pads_begin,
                                      CoordinateDiff& pads_end);
}  // namespace opset1
}  // namespace ngraph

using ngraph::get_constant_from_source;
