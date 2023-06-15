// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace element {

/**
 * @brief Defines supported elements for applying the element visitor action.
 *
 * @tparam List of supported ov::element types.
 *
 * The apply function check input element type, if is on the list apply function of visitor for specific type
 * if not found apply default action.
 */
template <Type_t...>
struct Supported;

/**
 * @brief Applies visitor action for not supported ov::element type.
 */
template <>
struct Supported<> {
    /**
     * @brief Applies visitor default action for not supported element type using Visitor default call operator.
     *
     * @tparam Visitor Type of element visitor functor.
     * @tparam Args    Type of visitor arguments used by visitor call operator.
     *
     * @param visitor  Visitor functor object.
     * @param args     Visitor arguments.
     * @return Value of result type returned by Visitor call operator.
     */
    template <class Visitor, class... Args>
    static auto apply(Type_t, Visitor&& visitor, Args&&... args) -> decltype(visitor(std::forward<Args>(args)...)) {
        return visitor(std::forward<Args>(args)...);
    }
};

/**
 * @brief Applies visitor action for supported element type defined by template parameters.
 *
 * @tparam ET      Current ov::element type used for check with input.
 * @tparam Others  Others supported ov::element.
 */
template <Type_t ET, Type_t... Others>
struct Supported<ET, Others...> {
    /**
     * @brief Applies visitor action for element type using Visitor call operator specified for by ET.
     *
     * @tparam Visitor Type of element visitor functor.
     * @tparam Args    Type of visitor arguments used by visitor call operator.
     *
     * @param et
     * @param visitor  Visitor functor object.
     * @param args     Visitor arguments.
     * @return Value of result type returned by Visitor call operator.
     */
    template <class Visitor, class... Args>
    static auto apply(Type_t et, Visitor&& visitor, Args&&... args)
        -> decltype(visitor.template operator()<ET>(std::forward<Args>(args)...)) {
        if (et == ET) {
            return visitor.template operator()<ET>(std::forward<Args>(args)...);
        } else {
            return Supported<Others...>::apply(et, std::forward<Visitor>(visitor), std::forward<Args>(args)...);
        }
    }
};

/**
 * @brief Helper functor which define no action for not supported type.
 *
 * @tparam R     Type of return value.
 * @tparam value Default value returned.
 */
template <class R, R... value>
struct NoAction {
    static_assert(sizeof...(value) < 2, "There should no more then one result value.");

    using result_type = R;

    template <class... Args>
    constexpr R operator()(Args&&...) const {
        return {value...};
    }
};

/**
 * @brief Helper functor which define no action for not supported type if result is void type.
 */
template <>
struct NoAction<void> {
    using result_type = void;

    template <class... Args>
    constexpr void operator()(Args&&...) const {}
};

/**
 * @brief Helper functor which throws ov::Exception for not supported element type.
 *
 * @tparam R Type of return type (used to be compatible with others call operator in Visitor).
 */
template <class R>
struct NotSupported {
    using result_type = R;

    template <class... Args>
    R operator()(Args&&...) {
        throw_not_supported();
        return static_cast<R>(NULL);
    }

private:
    [[noreturn]] static void throw_not_supported() {
        OPENVINO_THROW("Element not supported");
    }
};
}  // namespace element
}  // namespace ov
