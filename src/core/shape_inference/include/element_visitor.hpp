// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "openvino/core/constant_fold_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/itt.hpp"

namespace ov {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(ov_eval);
}  // namespace domains
}  // namespace itt

OV_ITT_DOMAIN(OV_PP_CAT(TYPE_LIST_, ov_eval));
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

#if defined(SELECTIVE_BUILD_ANALYZER)
    template <class Visitor, class... Args>
    static auto apply(const std::string& region, Type_t et, Args&&... args) -> typename Visitor::result_type {
        return Visitor::visit();
    }
#endif
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

#if defined(SELECTIVE_BUILD_ANALYZER)
    template <class Visitor, class... Args>
    static auto apply(const std::string& region, Type_t et, Args&&... args) -> typename Visitor::result_type {
        return (et == ET && is_cc_enabled<Visitor>(region))
                   ? Visitor::template visit<ET>(std::forward<Args>(args)...)
                   : IfTypeOf<Others...>::template apply<Visitor>(region, et, std::forward<Args>(args)...);
    }

    template <class Visitor>
    static bool is_cc_enabled(const std::string& region) {
        OV_ITT_SCOPED_TASK(OV_PP_CAT(TYPE_LIST_, ov_eval), region + "$" + Type(ET).to_string());
        return true;
    }
#endif
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

template <class... Args>
bool is_type_list_not_empty(Args&&... args) {
    return sizeof...(args) > 0;
}

}  // namespace element
}  // namespace ov

// Return ov::elements as parameter list e.g. OV_PP_ET_LIST(f16, i32) -> f16, i32
#define OV_PP_ET_LIST(...) OV_PP_EXPAND(__VA_ARGS__)

// Helpers to implement ignore or expand if symbol exists
#define OV_PP_ET_LIST_OR_EMPTY_0(...) OV_PP_IGNORE(__VA_ARGS__)
#define OV_PP_ET_LIST_OR_EMPTY_1(...) OV_PP_EXPAND(__VA_ARGS__)

// Check if ET list defined and use it for `IfTypeOf` class or make empty list
#define OV_PP_ET_LIST_OR_EMPTY(region)                                                                                \
    OV_PP_EXPAND(OV_PP_CAT(OV_PP_ET_LIST_OR_EMPTY_, OV_PP_IS_ENABLED(OV_PP_CAT(TYPE_LIST_ov_eval_enabled_, region)))( \
        OV_PP_CAT(TYPE_LIST_ov_eval_, region)))

/**
 * @brief Use this macro wrapper for ov::element::IfTypeOf class to integrate it with
 * OpenVINO conditional compilation feature.
 *
 * @param region  Region name for ITT which will be combined with TYPE_LIST_ prefix.
 * @param types   List ov::element IfTypeOf class e.g. OV_PP_ET_LIST(f16, i8) to pack as one paramater.
 * @param visitor Class name of visitor which will be used by IfTypeOf<types>::visit(_VA_ARGS_) function.
 * @param ...     List of parameters must match parameter list of `visit` function.
 *
 * @return Value returned by `visit` function
 */

#if defined(SELECTIVE_BUILD_ANALYZER)
#    define IF_TYPE_OF(region, types, visitor, ...) \
        ::ov::element::IfTypeOf<types>::apply<visitor>(OV_PP_TOSTRING(region), __VA_ARGS__)
#    define IF_TYPE_OF_CONVERT_TENSORS(region, node, outputs, inputs, types, visitor, ...)         \
        is_type_list_not_empty(types)                                                              \
            ? (::ov::element::IfTypeOf<types>::apply<visitor>(OV_PP_TOSTRING(region), __VA_ARGS__) \
                   ? true                                                                          \
                   : ov::util::evaluate_node_with_unsupported_precision(node, outputs, inputs))    \
            : false
#elif defined(SELECTIVE_BUILD)
#    define IF_TYPE_OF(region, types, visitor, ...) \
        ::ov::element::IfTypeOf<OV_PP_ET_LIST_OR_EMPTY(region)>::apply<visitor>(__VA_ARGS__)
#    define IF_TYPE_OF_CONVERT_TENSORS(region, node, outputs, inputs, types, visitor, ...)          \
        is_type_list_not_empty(types)                                                               \
            ? (::ov::element::IfTypeOf<OV_PP_ET_LIST_OR_EMPTY(region)>::apply<visitor>(__VA_ARGS__) \
                   ? true                                                                           \
                   : ov::util::evaluate_node_with_unsupported_precision(node, outputs, inputs))     \
            : false
#else
#    define IF_TYPE_OF(region, types, visitor, ...) ::ov::element::IfTypeOf<types>::apply<visitor>(__VA_ARGS__)
#    define IF_TYPE_OF_CONVERT_TENSORS(region, node, outputs, inputs, types, visitor, ...)      \
        is_type_list_not_empty(types)                                                           \
            ? (::ov::element::IfTypeOf<types>::apply<visitor>(__VA_ARGS__)                      \
                   ? true                                                                       \
                   : ov::util::evaluate_node_with_unsupported_precision(node, outputs, inputs)) \
            : false
#endif
