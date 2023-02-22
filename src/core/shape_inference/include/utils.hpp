// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <iterator>
#include <ngraph/validation_util.hpp>
#include <openvino/opsets/opset1.hpp>
#include <type_traits>

#include "bound_evaluation_util.hpp"
#include "shape_infer_type_utils.hpp"

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
/**
 * \brief Get the raw data as TResult object.
 *
 * \tparam T               TResult data type.
 * \tparam TResult         Type of return object, must support creation of std::inserter. Default std::vector<T>.
 * \tparam UnaryOperation  Unary function object applied on data with signature (T f(const U u)).
 *
 * \param et    Element type of input data.
 * \param ptr   Pointer to data of type et.
 * \param size  Data size as number of elements.
 * \param func  Unary operation function object.
 *
 * \throws ov::AssertionFailure for not supported element type.
 * \return Object of TResult with data from input pointer and transformed by unary operation.
 */
template <class T, class TResult = std::vector<T>, class UnaryOperation>
TResult get_raw_data_as(const element::Type_t et, const void* const ptr, const size_t size, UnaryOperation&& func) {
    OPENVINO_ASSERT(!!ptr, "ptr is Null");
    TResult out;
    auto out_it = std::inserter(out, out.end());

    switch (et) {
    case element::Type_t::i4: {
        using dtype = fundamental_type_for<element::Type_t::i4>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    case element::Type_t::i8: {
        using dtype = fundamental_type_for<element::Type_t::i8>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    case element::Type_t::i16: {
        using dtype = fundamental_type_for<element::Type_t::i16>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    case element::Type_t::i32: {
        using dtype = fundamental_type_for<element::Type_t::i32>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    case element::Type_t::i64: {
        using dtype = fundamental_type_for<element::Type_t::i64>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    case element::Type_t::u4: {
        using dtype = fundamental_type_for<element::Type_t::u4>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    case element::Type_t::u8: {
        using dtype = fundamental_type_for<element::Type_t::u8>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    case element::Type_t::u16: {
        using dtype = fundamental_type_for<element::Type_t::u16>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    case element::Type_t::u32: {
        using dtype = fundamental_type_for<element::Type_t::u32>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    case element::Type_t::u64: {
        using dtype = fundamental_type_for<element::Type_t::u64>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    case element::Type_t::f16: {
        using dtype = fundamental_type_for<element::Type_t::f16>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    case element::Type_t::f32: {
        using dtype = fundamental_type_for<element::Type_t::f32>;
        std::transform(static_cast<const dtype*>(ptr),
                       static_cast<const dtype*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    } break;
    default:
        OPENVINO_ASSERT(false, "Get raw data from tensor is not supported for element type: ", et);
    };
    return out;
}

/**
 * \brief Get data from Host tensor as object TResult.
 *
 * \tparam T               TResult data type.
 * \tparam TResult         Type of return object, must support creation of std::inserter. Default std::vector<T>.
 * \tparam UnaryOperation  Unary function object applied on data with signature (T f(const U u)).
 *
 * \param tv    Input host tensor.
 * \param func  Unary operation function object.
 *
 * \return Object of TResult with data from host tensor.
 */
template <class T, class TResult = std::vector<T>, class UnaryOperation>
TResult get_tensor_data_as(HostTensor& tv, UnaryOperation&& func) {
    auto t = Tensor(tv.get_element_type(), tv.get_shape(), tv.get_data_ptr());
    return get_tensor_data_as<T, TResult>(t, std::forward<UnaryOperation>(func));
}

template <class T, class TResult = std::vector<T>, class UnaryOperation>
TResult get_tensor_data_as(HostTensor* tv, UnaryOperation&& func) {
    return get_tensor_data_as<T, TResult>(*tv, std::forward<UnaryOperation>(func));
}

/**
 * \brief Get data from ov:tensor as object TResult.
 *
 * \tparam T               TResult data type.
 * \tparam TResult         Type of return object, must support creation of std::inserter. Default std::vector<T>.
 * \tparam UnaryOperation  Unary function object applied on data with signature (T f(const U u)).
 *
 * \param t     Input tensor.
 * \param func  Unary operation function object.
 *
 * \return Object of TResult with data from tensor.
 */
template <class T, class TResult = std::vector<T>, class UnaryOperation>
TResult get_tensor_data_as(const Tensor& t, UnaryOperation&& func) {
    return get_raw_data_as<T, TResult>(t.get_element_type(),
                                       t.data(),
                                       t.get_size(),
                                       std::forward<UnaryOperation>(func));
}

namespace op {
/**
 * \brief Get the operator's input const as pointer to vector of specified type.
 *
 * The behaviour depends on shape type. The default output type is std::vector<TData> can be replace by other type
 * which if is possible to construct it from constant data vector.
 *
 * \tparam TShape          Shape type which enabled this version (not ov::PartialShape)
 * \tparam TData           Type use to cast input's data.
 * \tparam TRes            Result type which has got default type as std::vector<TData>.
 * \tparam TTensorPtr      Type of tensor pointer or reference_wrapper. Default HostTensorPtr.
 * \tparam UnaryOperation  Unary function object applied on data with signature (Ret f(const TData &a)).
 *
 * \param op             Pointer to operator.
 * \param idx            Operator's input number.
 * \param constant_data  Map with constant. Default empty.
 * \param func           Unary operation function object.
 *
 * \return Pointer to constant data or nullptr if input has no constant data.
 */
template <class TShape,
          class TData,
          class TRes = std::vector<TData>,
          class TTensorPtr = HostTensorPtr,
          class UnaryOperation = ov::util::Cast<TData>,
          typename std::enable_if<!std::is_same<TShape, ov::PartialShape>::value>::type* = nullptr>
std::unique_ptr<TRes> get_input_const_data_as(const ov::Node* op,
                                              size_t idx,
                                              const std::map<size_t, TTensorPtr>& constant_data = {},
                                              UnaryOperation&& func = ov::util::Cast<TData>()) {
    if (constant_data.count(idx)) {
        return std::unique_ptr<TRes>(
            new TRes(get_tensor_data_as<TData, TRes>(constant_data.at(idx).get(), std::forward<UnaryOperation>(func))));
    } else {
        const auto& constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(idx));
        NODE_VALIDATION_CHECK(op, constant != nullptr, "Static shape inference lacks constant data on port ", idx);
        const auto& et = constant->get_element_type();
        const auto& shape = constant->get_shape();
        return std::unique_ptr<TRes>(new TRes(get_raw_data_as<TData, TRes>(et,
                                                                           constant->get_data_ptr(),
                                                                           shape_size(shape),
                                                                           std::forward<UnaryOperation>(func))));
    }
}

/**
 * \brief Get the operator's input const as pointer to vector of specified type.
 *
 * The behaviour depends on shape type. The default output type is std::vector<TData> can be replace by other type
 * which if is possible to construct it from constant data vector.
 *
 * \tparam TShape          Shape type which enabled this version (ov::PartialShape)
 * \tparam TData           Type use to cast input's data.
 * \tparam TRes            Result type which has got default type as std::vector<TData>.
 * \tparam TTensorPtr      Type of tensor pointer or reference_wrapper. Default HostTensorPtr.
 * \tparam UnaryOperation  Unary function object applied on data with signature (Ret f(const TData &a)).
 *
 * \param op             Pointer to operator.
 * \param idx            Operator's input number.
 * \param constant_data  Map with constant. Default empty.
 * \param func           Unary operation function object.
 *
 * \return Pointer to constant data or nullptr if input has no constant data.
 */
template <class TShape,
          class TData,
          class TRes = std::vector<TData>,
          class TTensorPtr = HostTensorPtr,
          class UnaryOperation = ov::util::Cast<TData>,
          typename std::enable_if<std::is_same<TShape, ov::PartialShape>::value>::type* = nullptr>
std::unique_ptr<TRes> get_input_const_data_as(const ov::Node* op,
                                              size_t idx,
                                              const std::map<size_t, TTensorPtr>& constant_data = {},
                                              UnaryOperation&& func = ov::util::Cast<TData>()) {
    if (constant_data.count(idx)) {
        return std::unique_ptr<TRes>(
            new TRes(get_tensor_data_as<TData, TRes>(constant_data.at(idx).get(), std::forward<UnaryOperation>(func))));
    } else if (const auto& constant = ov::get_constant_from_source(op->input_value(idx))) {
        const auto& et = constant->get_element_type();
        const auto& shape = constant->get_shape();
        return std::unique_ptr<TRes>(new TRes(get_raw_data_as<TData, TRes>(et,
                                                                           constant->get_data_ptr(),
                                                                           shape_size(shape),
                                                                           std::forward<UnaryOperation>(func))));
    } else {
        return {};
    }
}

/**
 * \brief Get the input const data as shape object.
 *
 * The input data can be processed by unary operation. By default is validated and casted to shape's dimension type.
 *
 * \tparam TShape
 * \tparam TTensorPtr      Type of tensor pointer or reference_wrapper. Default HostTensorPtr.
 * \tparam UnaryOperation  Unary function object applied on data with signature (Ret f(const TDimValue &a)).
 *
 * \param op             Pointer to operator.
 * \param idx            Operator input index.
 * \param constant_data  Map with constant data. Default empty.
 * \param func           Unary operation function object to apply in input data.
 *                       Default ov::utils::InTypeRange<TDimValue>.
 *
 * \return Unique pointer to shape created from input data.
 */
template <class TShape,
          class TDimValue = typename TShape::value_type::value_type,
          class TTensorPtr = HostTensorPtr,
          class UnaryOperation = ov::util::InTypeRange<TDimValue>>
std::unique_ptr<TShape> get_input_const_data_as_shape(const ov::Node* op,
                                                      size_t idx,
                                                      const std::map<size_t, TTensorPtr>& constant_data = {},
                                                      UnaryOperation&& func = ov::util::InTypeRange<TDimValue>()) {
    std::unique_ptr<TShape> shape_ptr;

    if (auto d =
            get_input_const_data_as<TShape, TDimValue>(op, idx, constant_data, std::forward<UnaryOperation>(func))) {
        return std::unique_ptr<TShape>(new TShape(std::move(*d)));
    } else {
        PartialShape shape;
        if (ov::evaluate_as_partial_shape(op->input_value(idx), shape)) {
            return std::unique_ptr<TShape>(new TShape(std::move(shape)));
        }
    }
    return {};
}

/**
 * \brief Get the input bounds from constant input (constant map) or evaluate bunds
 *  and return them as vector of pairs (lower, upper).
 *
 * \tparam TShape        Shape type.
 * \tparam TData         Bound value type.
 *
 * \param op             Operator pointer.
 * \param idx            Input index.
 * \param constant_data  Map with constant data.
 *
 * \return Return vector of bounds as pair lower, upper.
 */
template <class TShape, class TData, class TResult = std::vector<std::pair<TData, TData>>>
std::unique_ptr<TResult> get_input_bounds(const ov::Node* op,
                                          size_t idx,
                                          const std::map<size_t, HostTensorPtr>& constant_data) {
    const auto make_bound = [](TData lb, TData ub) -> typename TResult::value_type {
        return {lb, ub};
    };

    if (auto lowers = op::get_input_const_data_as<TShape, TData>(op, idx, constant_data)) {
        auto out = std::unique_ptr<TResult>(new TResult);
        out->reserve(lowers->size());
        std::transform(lowers->begin(), lowers->end(), lowers->begin(), std::back_inserter(*out), make_bound);
        return out;
    } else {
        auto bounds = ov::evaluate_both_bounds(op->get_input_source_output(idx));

        if (bounds.first && bounds.second) {
            constexpr auto cast = ov::util::Cast<TData>();
            auto lowers = get_tensor_data_as<TData>(bounds.first, cast);
            auto uppers = get_tensor_data_as<TData>(bounds.second, cast);

            auto out = std::unique_ptr<TResult>(new TResult);
            out->reserve(lowers.size());
            std::transform(lowers.begin(), lowers.end(), uppers.begin(), std::back_inserter(*out), make_bound);
            return out;
        }
    }
    return {};
}

}  // namespace op
}  // namespace ov

// Helper to reduce duplicates of code for get_data_as_... specific type functions.
template <class TShape, class TData>
inline bool get_data_as(const ov::Node* op,
                        size_t idx,
                        std::vector<TData>& data_out,
                        const std::map<size_t, ov::HostTensorPtr>& constant_data = {}) {
    if (auto out = ov::op::get_input_const_data_as<TShape, TData>(op, idx, constant_data, ov::util::Cast<TData>())) {
        data_out = std::move(*out);
        return true;
    } else {
        return false;
    }
}

template <class TShape>
inline bool get_data_as_int64(size_t idx,
                              const ov::Node* op,
                              std::vector<int64_t>& axes_value,
                              const std::map<size_t, ov::HostTensorPtr>& constant_data = {}) {
    return get_data_as<TShape>(op, idx, axes_value, constant_data);
}

template <class TShape>
inline bool get_data_as_float(size_t idx,
                              const ov::Node* op,
                              std::vector<float>& axes_value,
                              const std::map<size_t, ov::HostTensorPtr>& constant_data = {}) {
    return get_data_as<TShape>(op, idx, axes_value, constant_data);
}

/**
 * \brief Get the operator's constant data as shape of type T.
 *
 *  \note The constant data are get as size_t (Dimension value type for static shape). If pointed input is signed the
 *  output shape dimension can be wrongly interpreted.
 *
 * \tparam TShape        Shape type.
 *
 * \param idx            Operator's input index.
 * \param op             Pointer to operator.
 * \param shape          Output shape made from constant data.
 * \param constant_data  Map with constant tensors. Optional default empty.
 *
 * \return true If constant data acquired as shape otherwise throws NodeValidation exception.
 */
template <class TShape>
inline bool get_data_as_shape(size_t idx,
                              const ov::Node* op,
                              TShape& shape,
                              const std::map<size_t, ov::HostTensorPtr>& constant_data = {}) {
    using TDimValue = typename TShape::value_type::value_type;
    shape =
        std::move(*ov::op::get_input_const_data_as_shape<TShape>(op, idx, constant_data, ov::util::Cast<TDimValue>()));
    return true;
}

/**
 * \brief Get the operator's constant data as ov::PartialShape.
 *
 * If data not get as constant then try evaluate this input as partial shape from input's bounds  and labels.
 *
 *  \note The constant data are get as int64_t. If pointed input is unsigned then output shape
 *  dimension can be wrongly interpreted.
 *
 * \param idx            Operator's input index.
 * \param op             Pointer to operator.
 * \param shape          Output shape made from constant data.
 * \param constant_data  Map with constant tensors. Optional default empty.
 *
 * \return true If constant data acquired as shape otherwise throws NodeValidation exception.
 */
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
