// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <ngraph/validation_util.hpp>
#include <openvino/opsets/opset1.hpp>

template <class OpType, class T>
void copy_shape_infer(const OpType* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() == 1 && output_shapes.size() == 1,
                          "Incorrect number of input/output shapes");
    output_shapes[0] = input_shapes[0];
}

template <class OpType, class T>
void first_input_passthrough_infer(const OpType* op,
                                   const std::vector<T>& input_shapes,
                                   std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op,
                          output_shapes.size() == 1 && input_shapes.size() >= 1,
                          "Incorrect number of input and output shapes");
    output_shapes[0] = input_shapes[0];
}

template <class OpType, class T>
void eltwise_shape_infer(const OpType* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() == 2 && output_shapes.size() == 1,
                          "Incorrect number of input/output shapes");
    auto output_shape = input_shapes[0];
    const auto& autob = op->get_autob();
    if (autob.m_type == ov::op::AutoBroadcastType::NONE) {
        NODE_VALIDATION_CHECK(op, T::merge_into(output_shape, input_shapes[1]), "Argument shapes are inconsistent.");
    } else if (autob.m_type == ov::op::AutoBroadcastType::NUMPY || autob.m_type == ov::op::AutoBroadcastType::PDPD) {
        NODE_VALIDATION_CHECK(op,
                              T::broadcast_merge_into(output_shape, input_shapes[1], autob),
                              "Argument shapes are inconsistent.");
    } else {
        NODE_VALIDATION_CHECK(op, false, "Unsupported auto broadcast specification");
    }
    output_shapes[0] = output_shape;
}

namespace ov {
namespace op {

/**
 * \brief Get the operator's input const as pointer to vector of specified type.
 *
 * The behaviour depends on shape type. The default output type is std::vector<TData> can be replace by other type
 * which if is possible to construct it from constant data vector.
 *
 * \tparam TShape  Shape type which enabled this version (not ov::PartialShape)
 * \tparam TData   Type use to cast input's data.
 * \tparam TRes    Result type which has got default type as std::vector<TData>.
 *
 * \param op             Pointer to operator.
 * \param idx            Operator's input number.
 * \param constant_data  Map with constant. Default empty.
 *
 * \return Pointer to constant data or nullptr if input has no constant data.
 */
template <class TShape,
          class TData,
          class TRes = std::vector<TData>,
          typename std::enable_if<!std::is_same<TShape, ov::PartialShape>::value>::type* = nullptr>
std::unique_ptr<TRes> get_input_const_data_as(const ov::Node* op,
                                              size_t idx,
                                              const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    if (constant_data.count(idx)) {
        return std::unique_ptr<TRes>(new TRes(ov::opset1::Constant(constant_data.at(idx)).cast_vector<TData>()));
    } else {
        const auto& constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(idx));
        NODE_VALIDATION_CHECK(op, constant != nullptr, "Static shape inference lacks constant data on port ", idx);
        return std::unique_ptr<TRes>(new TRes(constant->cast_vector<TData>()));
    }
}

/**
 * \brief Get the operator's input const as pointer to vector of specified type.
 *
 * The behaviour depends on shape type. The default output type is std::vector<TData> can be replace by other type
 * which if is possible to construct it from constant data vector.
 *
 * \tparam TShape  Shape type which enabled this version (ov::PartialShape)
 * \tparam TData   Type use to cast input's data.
 * \tparam TRes    Result type which has got default type as std::vector<TData>.
 *
 * \param op             Pointer to operator.
 * \param idx            Operator's input number.
 * \param constant_data  Map with constant. Default empty.
 *
 * \return Pointer to constant data or nullptr if input has no constant data.
 */
template <class TShape,
          class TData,
          class TRes = std::vector<TData>,
          typename std::enable_if<std::is_same<TShape, ov::PartialShape>::value>::type* = nullptr>
std::unique_ptr<std::vector<TData>> get_input_const_data_as(const ov::Node* op,
                                                            size_t idx,
                                                            const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    if (constant_data.count(idx)) {
        return std::unique_ptr<TRes>(new TRes(ov::opset1::Constant(constant_data.at(idx)).cast_vector<TData>()));
    } else if (const auto& constant = ov::get_constant_from_source(op->input_value(idx))) {
        return std::unique_ptr<TRes>(new TRes(constant->cast_vector<TData>()));
    } else {
        return {};
    }
}

}  // namespace op
}  // namespace ov

template <class T>
inline bool get_data_as_int64(
    size_t idx,
    const ov::Node* op,
    std::vector<int64_t>& axes_value,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    if (constant_data.count(idx)) {
        axes_value = ov::opset1::Constant(constant_data.at(idx)).cast_vector<int64_t>();
    } else {
        const auto& constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(idx));
        NODE_VALIDATION_CHECK(op, constant != nullptr, "Static shape inference lacks constant data on port ", idx);
        axes_value = constant->cast_vector<int64_t>();
    }
    return true;
}

template <>
inline bool get_data_as_int64<ov::PartialShape>(
    size_t idx,
    const ov::Node* op,
    std::vector<int64_t>& axes_value,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (constant_data.count(idx)) {
        axes_value = ov::opset1::Constant(constant_data.at(idx)).cast_vector<int64_t>();
    } else if (const auto& constant = ov::get_constant_from_source(op->input_value(idx))) {
        axes_value = constant->cast_vector<int64_t>();
    } else {
        return false;
    }
    return true;
}

template <class T>
inline bool get_data_as_float(
    size_t idx,
    const ov::Node* op,
    std::vector<float>& axes_value,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    if (constant_data.count(idx)) {
        axes_value = ov::opset1::Constant(constant_data.at(idx)).cast_vector<float>();
    } else {
        const auto& constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(idx));
        NODE_VALIDATION_CHECK(op, constant != nullptr, "Static shape inference lacks constant data on port ", idx);
        axes_value = constant->cast_vector<float>();
    }
    return true;
}

template <>
inline bool get_data_as_float<ov::PartialShape>(
    size_t idx,
    const ov::Node* op,
    std::vector<float>& axes_value,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (constant_data.count(idx)) {
        axes_value = ov::opset1::Constant(constant_data.at(idx)).cast_vector<float>();
    } else if (const auto& constant = ov::get_constant_from_source(op->input_value(idx))) {
        axes_value = constant->cast_vector<float>();
    } else {
        return false;
    }
    return true;
}

template <class T>
inline bool get_data_as_shape(
    size_t idx,
    const ov::Node* op,
    T& shape,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    if (constant_data.count(idx)) {
        shape = T(ov::opset1::Constant(constant_data.at(idx)).cast_vector<size_t>());
    } else {
        const auto& constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(idx));
        NODE_VALIDATION_CHECK(op, constant != nullptr, "Static shape inference lacks constant data on port ", idx);
        shape = T(constant->cast_vector<size_t>());
    }
    return true;
}

template <>
inline bool get_data_as_shape<ov::PartialShape>(
    size_t idx,
    const ov::Node* op,
    ov::PartialShape& shape,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (constant_data.count(idx)) {
        shape = ov::PartialShape(ov::opset1::Constant(constant_data.at(idx)).cast_vector<int64_t>());
        return true;
    } else {
        return ov::evaluate_as_partial_shape(op->input_value(idx), shape);
    }
}

template <class T>
inline void check_divided_result(const ov::Node* op,
                                 const T& res,
                                 const T& divided,
                                 const typename T::value_type& divisor) {
    NODE_VALIDATION_CHECK(op,
                          res != T{},
                          "Dimension value: [ ",
                          divided.get_min_length(),
                          ", ",
                          divided.get_max_length(),
                          "]",
                          " must be a multiple of divisor: ",
                          divisor);
}

template <>
inline void check_divided_result<ov::Dimension>(const ov::Node* op,
                                                const ov::Dimension& res,
                                                const ov::Dimension& divided,
                                                const typename ov::Dimension::value_type& divisor) {
    NODE_VALIDATION_CHECK(op,
                          !res.get_interval().empty(),
                          "Dimension value: [ ",
                          divided.get_min_length(),
                          ", ",
                          divided.get_max_length(),
                          "]",
                          " must be a multiple of divisor: ",
                          divisor);
}
