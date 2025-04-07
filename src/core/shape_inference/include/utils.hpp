// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <iterator>
#include <optional>
#include <type_traits>

#include "element_visitor.hpp"
#include "openvino/core/bound_evaluation_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset1.hpp"
#include "shape_infer_type_utils.hpp"
#include "tensor_data_accessor.hpp"

namespace ov {

struct TensorTransform : element::NotSupported<void> {
    using element::NotSupported<void>::visit;

    template <element::Type_t ET, class Iterator, class UnaryOperation>
    static result_type visit(const void* const ptr, const size_t size, Iterator out_it, UnaryOperation&& func) {
        using T = fundamental_type_for<ET>;
        std::transform(static_cast<const T*>(ptr),
                       static_cast<const T*>(ptr) + size,
                       out_it,
                       std::forward<UnaryOperation>(func));
    }
};

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

    using namespace ov::element;
    IfTypeOf<bf16, f16, f32, f64, i4, i8, i16, i32, i64, u4, u8, u16, u32, u64, nf4>::apply<TensorTransform>(
        et,
        ptr,
        size,
        out_it,
        std::forward<UnaryOperation>(func));
    return out;
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
template <class T, class TResult = std::vector<T>, class UnaryOperation = ov::util::Cast<T>>
TResult get_tensor_data_as(const Tensor& t, UnaryOperation&& func = ov::util::Cast<T>()) {
    return get_raw_data_as<T, TResult>(t.get_element_type(),
                                       t.data(),
                                       t.get_size(),
                                       std::forward<UnaryOperation>(func));
}

namespace util {
/**
 * \brief Check if value of type T has got maximum value of type U.
 *
 * \tparam T     Input value type
 * \tparam U     Type to get its minimum for comparision. Default same as T.
 *
 * \param value  Input value.
 *
 * \return       True if input value has got maximum value of type U otherwise false.
 */
template <class T, class U = T>
constexpr bool is_max(const T& value) {
    return std::numeric_limits<U>::max() == value;
}

/**
 * \brief Check if value of type T has got minimum value of type U.
 *
 * \tparam T     Input value type.
 * \tparam U     Type to get its minimum for comparision. Default same as T.
 *
 * \param value  Input value.
 *
 * \return       True if input value has got minimum value of type U otherwise false.
 */
template <class T, class U = T>
constexpr bool is_min(const T& value) {
    return std::numeric_limits<U>::min() == value;
}
}  // namespace util

namespace element {
/**
 * \brief  Check if value has got maximum value of ov::element::Type_t
 *
 * \tparam T     Input value type.
 *
 * \param type   ov::element type to get its maximum.
 * \param value  Input value for check.
 *
 * \return True if input value has got maximum number specified by ov::element type otherwise false.
 */
template <class T>
bool is_max_of(const element::Type_t& type, const T& value) {
    switch (type) {
    case element::i32:
        return util::is_max<T, typename element_type_traits<element::i32>::value_type>(value);
    case element::i64:
        return util::is_max<T, typename element_type_traits<element::i64>::value_type>(value);
    default:
        return false;
    }
}

/**
 * \brief  Check if value has got minimum value of ov::element::Type_t
 *
 * \tparam T     Input value type.
 *
 * \param type   ov::element type to get its minimum.
 * \param value  Input value for check.
 *
 * \return True if input value has got minimum number specified by ov::element type otherwise false.
 */
template <class T>
bool is_min_of(const element::Type_t type, const T& value) {
    switch (type) {
    case element::i32:
        return util::is_min<T, typename element_type_traits<element::i32>::value_type>(value);
    case element::i64:
        return util::is_min<T, typename element_type_traits<element::i64>::value_type>(value);
    default:
        return false;
    }
}

/**
 * \brief  Checks input value for element type maximum or minimum and return limit or value.
 *
 * \tparam T     Type of input value.
 * \tparam U     Type of return value. Default same as T.
 *
 * \param type   Type of ov::element::Type_t
 * \param value  Input value for check.
 *
 * \return If value is maximum or minimum get limit of U otherwise value as U.
 */
template <class T, class U = T>
U get_value_or_limit_of(const element::Type_t& type, const T& value) {
    if (is_min_of(type, value)) {
        return std::numeric_limits<U>::min();
    } else if (is_max_of(type, value)) {
        return std::numeric_limits<U>::max();
    } else {
        return static_cast<U>(value);
    }
}
}  // namespace element

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
 * \tparam UnaryOperation  Unary function object applied on data with signature (Ret f(const TData &a)).
 *
 * \param op               Pointer to operator.
 * \param idx              Operator's input number.
 * \param tensor_accessor  Tensor accessor object.
 * \param func             Unary operation function object.
 *
 * \return Pointer to constant data or nullptr if input has no constant data.
 */
template <class TShape,
          class TData,
          class TRes = std::vector<TData>,
          class UnaryOperation = ov::util::Cast<TData>,
          typename std::enable_if<!std::is_same<TShape, ov::PartialShape>::value>::type* = nullptr>
std::optional<TRes> get_input_const_data_as(const ov::Node* op,
                                            size_t idx,
                                            const ITensorAccessor& tensor_accessor,
                                            UnaryOperation&& func = ov::util::Cast<TData>()) {
    if (auto t = tensor_accessor(idx)) {
        return {get_tensor_data_as<TData, TRes>(t, std::forward<UnaryOperation>(func))};
    } else {
        const auto& constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(idx));
        NODE_VALIDATION_CHECK(op, constant != nullptr, "Static shape inference lacks constant data on port ", idx);
        const auto& et = constant->get_element_type();
        const auto& shape = constant->get_shape();
        return {get_raw_data_as<TData, TRes>(et,
                                             constant->get_data_ptr(),
                                             shape_size(shape),
                                             std::forward<UnaryOperation>(func))};
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
 * \tparam UnaryOperation  Unary function object applied on data with signature (Ret f(const TData &a)).
 *
 * \param op               Pointer to operator.
 * \param idx              Operator's input number.
 * \param tensor_accessor  Tensor accessor object.
 * \param func             Unary operation function object.
 *
 * \return Pointer to constant data or nullptr if input has no constant data.
 */
template <class TShape,
          class TData,
          class TRes = std::vector<TData>,
          class UnaryOperation = ov::util::Cast<TData>,
          typename std::enable_if<std::is_same<TShape, ov::PartialShape>::value>::type* = nullptr>
std::optional<TRes> get_input_const_data_as(const ov::Node* op,
                                            size_t idx,
                                            const ITensorAccessor& tensor_accessor,
                                            UnaryOperation&& func = ov::util::Cast<TData>()) {
    if (auto t = tensor_accessor(idx)) {
        return {get_tensor_data_as<TData, TRes>(t, std::forward<UnaryOperation>(func))};
    } else if (const auto& constant =
                   (idx < op->get_input_size()) ? ov::util::get_constant_from_source(op->input_value(idx)) : nullptr) {
        const auto& et = constant->get_element_type();
        const auto& shape = constant->get_shape();
        return {get_raw_data_as<TData, TRes>(et,
                                             constant->get_data_ptr(),
                                             shape_size(shape),
                                             std::forward<UnaryOperation>(func))};
    } else {
        return {};
    }
}

/**
 * \brief Get the input const data as shape object.
 *
 * The input data can be processed by unary operation. By default is validated and casted to shape's dimension type.
 *
 * \tparam TShape          Shape type.
 * \tparam TDimValue       Dimension value type.
 * \tparam UnaryOperation  Unary function object applied on data with signature (Ret f(const TDimValue &a)).
 *
 * \param op               Pointer to operator.
 * \param port             Input port number.
 * \param tensor_accessor  Tensor accessor object.
 * \param func             Unary operation function object to apply in input data.
 *                         Default ov::utils::InTypeRange<TDimValue>.
 *
 * \return Unique pointer to shape created from input data.
 */
template <class TShape,
          class TDimValue = typename TShape::value_type::value_type,
          class UnaryOperation = ov::util::InTypeRange<TDimValue>,
          typename std::enable_if<!std::is_same<TShape, ov::PartialShape>::value>::type* = nullptr>
std::optional<TShape> get_input_const_data_as_shape(const ov::Node* op,
                                                    size_t port,
                                                    const ITensorAccessor& tensor_accessor,
                                                    UnaryOperation&& func = ov::util::InTypeRange<TDimValue>()) {
    auto shape = std::optional<TShape>();
    if (auto s = get_input_const_data_as<TShape, TDimValue, TShape>(op,
                                                                    port,
                                                                    tensor_accessor,
                                                                    std::forward<UnaryOperation>(func))) {
        shape = std::move(*s);
    }
    return shape;
}

template <class TShape,
          class TDimValue = typename TShape::value_type::value_type,
          class UnaryOperation = ov::util::InTypeRange<TDimValue>,
          typename std::enable_if<std::is_same<TShape, ov::PartialShape>::value>::type* = nullptr>
std::optional<TShape> get_input_const_data_as_shape(const ov::Node* op,
                                                    size_t port,
                                                    const ITensorAccessor& tensor_accessor,
                                                    UnaryOperation&& func = ov::util::InTypeRange<TDimValue>()) {
    auto shape = std::optional<TShape>();
    if (auto t = tensor_accessor(port)) {
        shape.emplace(get_tensor_data_as<TDimValue>(t, std::forward<UnaryOperation>(func)));
    } else if (port < op->get_input_size()) {
        PartialShape s;
        if (auto c = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(port))) {
            shape.emplace(get_raw_data_as<TDimValue>(c->get_element_type(),
                                                     c->get_data_ptr(),
                                                     shape_size(c->get_shape()),
                                                     std::forward<UnaryOperation>(func)));
        } else if (ov::util::evaluate_as_partial_shape(op->input_value(port), s)) {
            shape = std::move(s);
        }
    }
    return shape;
}

// To get element type from constant or tensor.
inline element::Type get_input_const_element_type(const ov::Node* const op, size_t port, const ITensorAccessor& ta) {
    if (auto t = ta(port)) {
        return t.get_element_type();
    } else if (const auto& constant = ov::util::get_constant_from_source(op->input_value(port))) {
        return constant->get_element_type();
    } else {
        return element::dynamic;
    }
}

/**
 * \brief Get the input bounds from constant input or try evaluate bunds
 *  and return them as vector of pairs (lower, upper).
 *
 * \tparam TShape        Shape type.
 * \tparam TData         Bound value type.
 *
 * \param op    Operator pointer.
 * \param port  Input port number.
 * \param ta    Tensor accessor to constant data.
 *
 * \return Return optional vector of bounds as pair lower, upper when evaluated successful.
 */
template <class TShape, class TData, class TResult = std::vector<std::pair<TData, TData>>>
std::optional<TResult> get_input_bounds(const ov::Node* op, size_t port, const ITensorAccessor& ta) {
    const auto make_bound = [](element::Type_t et) {
        return [et](TData lb, TData ub) -> typename TResult::value_type {
            return {element::get_value_or_limit_of(et, lb), element::get_value_or_limit_of(et, ub)};
        };
    };

    constexpr auto cast = ov::util::Cast<TData>();
    std::optional<TResult> out;

    if (const auto t = ta(port)) {
        const auto& et = t.get_element_type();
        const auto lowers = get_tensor_data_as<TData>(t, cast);
        out.emplace();
        out->reserve(lowers.size());
        std::transform(lowers.cbegin(), lowers.cend(), lowers.cbegin(), std::back_inserter(*out), make_bound(et));
    } else if (port < op->get_input_size()) {
        auto bounds = ov::util::evaluate_both_bounds(op->get_input_source_output(port));

        if (bounds.first && bounds.second) {
            const auto& et = bounds.first.get_element_type();
            auto lowers = get_tensor_data_as<TData>(bounds.first, cast);
            auto uppers = get_tensor_data_as<TData>(bounds.second, cast);

            out.emplace();
            out->reserve(lowers.size());
            std::transform(lowers.begin(), lowers.end(), uppers.begin(), std::back_inserter(*out), make_bound(et));
        }
    }

    if (!std::is_same<TShape, PartialShape>::value) {
        NODE_VALIDATION_CHECK(op, out, "Static shape inference lacks constant data on port ", port);
    }
    return out;
}

/**
 * @brief Inference broadcast shape for element wise operator according to broadcast specification stored in operator.
 *
 * @param op      Pointer to operator.
 * @param first   First input shape.
 * @param second  Second input shape.
 *
 * @return Result shape from inputs with applied broadcast specification.
 */
ov::Shape infer_broadcast_shape(const ov::Node* const op, const ov::Shape& first, const ov::Shape& second);

/**
 * @brief Inference broadcast shape from input tensor shapes for element wise operator
 * according to broadcast specification stored in operator.
 *
 * @param op      Pointer to operator.
 * @param inputs  Tensors vector to get theirs shapes.
 *
 * @return Result shape from input tensors shape with applied broadcast specification.
 */
ov::Shape infer_broadcast_shape(const ov::Node* const op, const ov::TensorVector& inputs);
}  // namespace op

/**
 * @brief Get correct return type of input shape when call `shape_infer`.
 *
 * The input shapes are vector like std::vector<TShape>, where `TShape` can be `std::vector<const size_t>`
 * This will provide correct return especially for static shape which can work as reference to dimension or hold them.
 *
 * @tparam TShape Type of input shape.
 */
template <class TShape>
struct result_shape {
    using type = typename TShape::ShapeContainer;
};

/**
 * @brief Get correct result shape for PartialShape which is same type.
 */
template <>
struct result_shape<PartialShape> {
    using type = PartialShape;
};

/**
 * @brief Get correct result shape for ov::Shape which is same type.
 */
template <>
struct result_shape<ov::Shape> {
    using type = ov::Shape;
};

template <class TShape>
using result_shape_t = typename result_shape<TShape>::type;
}  // namespace ov

/**
 * @brief Check for valid quotient of dimension division.
 *
 * If quotient is not valid (quotient * divisor != dividend) throw NodeValidationFailure exception.
 *
 * @tparam TDim     Type of dimension.
 *
 * @param op        Pointer to operator.
 * @param quotient  Dimension result after division.
 * @param dividend  Original dimension.
 * @param divisor   Dimension divide value.
 */
template <class TDim>
inline void check_divided_result(const ov::Node* op,
                                 const TDim& quotient,
                                 const TDim& dividend,
                                 const typename TDim::value_type& divisor) {
    NODE_VALIDATION_CHECK(op,
                          quotient != TDim{},
                          "Dimension value: [ ",
                          dividend.get_min_length(),
                          ", ",
                          dividend.get_max_length(),
                          "]",
                          " must be a multiple of divisor: ",
                          divisor);
}

template <>
inline void check_divided_result<ov::Dimension>(const ov::Node* op,
                                                const ov::Dimension& quotient,
                                                const ov::Dimension& dividend,
                                                const typename ov::Dimension::value_type& divisor) {
    NODE_VALIDATION_CHECK(op,
                          !quotient.get_interval().empty(),
                          "Dimension value: [ ",
                          dividend.get_min_length(),
                          ", ",
                          dividend.get_max_length(),
                          "]",
                          " must be a multiple of divisor: ",
                          divisor);
}
