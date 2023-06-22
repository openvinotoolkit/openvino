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
struct IfTypeOf;

/**
 * @brief Applies visitor action for not supported ov::element type.
 */
template <>
struct IfTypeOf<> {
    /**
     * @brief Applies visitor default action for not supported element type using Visitor non-template visit function.
     *
     * @tparam Visitor Visitor class implementing visit function.
     * @tparam Args    Types of visit parameters.
     *
     * @param et       Input element type.
     * @param args     Visitor arguments.
     * @return Value of result type returned by Visitor.
     */
    template <class Visitor, class... Args>
    static auto apply(Type_t et, Args&&... args) -> typename Visitor::result_type {
        return Visitor::visit();
    }
};

/**
 * @brief Applies visitor action for supported element type defined by template parameters.
 *
 * @tparam ET      Current ov::element type used for check with input.
 * @tparam Others  Others supported ov::element.
 */
template <Type_t ET, Type_t... Others>
struct IfTypeOf<ET, Others...> {
    /**
     * @brief Applies visitor action for element type using Visitor visit function for ET.
     *
     * @tparam Visitor Visitor class implementing visit function.
     * @tparam Args    Types of visit parameters.
     *
     * @param et       Input element type.
     * @param args     Visitor arguments.
     * @return Value of result type returned by Visitor.
     */
    template <class Visitor, class... Args>
    static auto apply(Type_t et, Args&&... args) -> typename Visitor::result_type {
        return (et == ET) ? Visitor::template visit<ET>(std::forward<Args>(args)...)
                          : IfTypeOf<Others...>::template apply<Visitor>(et, std::forward<Args>(args)...);
    }
};

/**
 * @brief Helper visitor which define no action for not supported type.
 *
 * @tparam R     Type of return value.
 * @tparam value Default value returned.
 */
template <class R, R... value>
struct NoAction {
    static_assert(sizeof...(value) < 2, "There should no more then one result value.");

    using result_type = R;

    static constexpr R visit() {
        return {value...};
    }
};

/**
 * @brief Helper visitor which define no action for not supported type if result is void type.
 */
template <>
struct NoAction<void> {
    using result_type = void;

    static void visit() {}
};

/**
 * @brief Helper visitor which throws ov::Exception for not supported element type.
 *
 * @tparam R Type of return type (used to be compatible with others call operator in Visitor).
 */
template <class R>
struct NotSupported {
    using result_type = R;

    [[noreturn]] static R visit() {
        throw_not_supported();
    }

private:
    [[noreturn]] static void throw_not_supported() {
        OPENVINO_THROW("Element not supported");
    }
};
}  // namespace element
}  // namespace ov
