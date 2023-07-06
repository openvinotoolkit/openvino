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
 * @brief Primary template defines suppoted element types.
 *
 * The list of element types is used to check if runtime value of element type is one in the list.
 * Base on this check the Visitor::visit function is called for specific element type.
 *
 * @tparam List of supported ov::element types.
 */
template <Type_t...>
struct IfTypeOf;

/**
 * @brief Applies visitor action for not supported ov::element type.
 */
template <>
struct IfTypeOf<> {
    /**
     * @brief Applies visitor default action if input element type is not not supported by IfTypeOf.
     *
     * Uses Visitor::visit non-template function.
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
     * @brief Applies visitor action if input element type is same as ET.
     *
     * Uses Visitor::visit<ET> function if `et == ET`, otherwise check input element type against Others.
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
 * @brief Helper visitor which defines no action for not supported type.
 *
 * @tparam R     Type of return value.
 * @tparam value Default value returned.
 */
template <class R, R... value>
struct NoAction {
    static_assert(sizeof...(value) < 2, "There should be no more than one result value.");

    using result_type = R;

    static constexpr result_type visit() {
        return {value...};
    }
};

/**
 * @brief Helper visitor which defines no action for not supported type if result is void type.
 */
template <>
struct NoAction<void> {
    using result_type = void;

    static result_type visit() {}
};

/**
 * @brief Helper visitor which throws ov::Exception for not supported element type.
 *
 * @tparam R  Type of return value.
 */
template <class R>
struct NotSupported {
    using result_type = R;

    [[noreturn]] static result_type visit() {
        throw_not_supported();
    }

private:
    [[noreturn]] static void throw_not_supported() {
        OPENVINO_THROW("Element not supported");
    }
};
}  // namespace element
}  // namespace ov
