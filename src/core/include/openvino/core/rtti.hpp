// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type.hpp"
#include "openvino/core/visibility.hpp"

#define _OPENVINO_RTTI_EXPAND(X)                                    X
#define _OPENVINO_RTTI_DEFINITION_SELECTOR_2(_1, _2, NAME, ...)     NAME
#define _OPENVINO_RTTI_DEFINITION_SELECTOR_3(_1, _2, _3, NAME, ...) NAME

#define _OPENVINO_RTTI_WITH_TYPE(TYPE_NAME) _OPENVINO_RTTI_WITH_TYPE_VERSION(TYPE_NAME, "util")

#define _OPENVINO_RTTI_WITH_TYPE_VERSION(TYPE_NAME, VERSION_NAME)                         \
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() { \
        static ::ov::DiscreteTypeInfo type_info_static{TYPE_NAME, VERSION_NAME};          \
        type_info_static.hash();                                                          \
        return type_info_static;                                                          \
    }                                                                                     \
    const ::ov::DiscreteTypeInfo& get_type_info() const override { return get_type_info_static(); }

#define _OPENVINO_RTTI_WITH_TYPE_VERSION_PARENT(TYPE_NAME, VERSION_NAME, PARENT_CLASS) \
    _OPENVINO_RTTI_WITH_TYPE_VERSIONS_PARENT(TYPE_NAME, VERSION_NAME, PARENT_CLASS)

#define _OPENVINO_RTTI_WITH_TYPE_VERSIONS_PARENT(TYPE_NAME, VERSION_NAME, PARENT_CLASS)        \
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() {      \
        static ::ov::DiscreteTypeInfo type_info_static{TYPE_NAME,                              \
                                                       VERSION_NAME,                           \
                                                       &PARENT_CLASS::get_type_info_static()}; \
        type_info_static.hash();                                                               \
        return type_info_static;                                                               \
    }                                                                                          \
    const ::ov::DiscreteTypeInfo& get_type_info() const override { return get_type_info_static(); }

/// Helper macro that puts necessary declarations of RTTI block inside a class definition.
/// Should be used in the scope of class that requires type identification besides one provided by
/// C++ RTTI.
/// Recommended to be used for all classes that are inherited from class ov::Node to enable
/// pattern
/// matching for them. Accepts necessary type identification details like type of the operation,
/// version and optional parent class.
///
/// Applying this macro within a class definition provides declaration of type_info static
/// constant for backward compatibility with old RTTI definition for Node,
/// static function get_type_info_static which returns a reference to an object that is equal to
/// type_info but not necessary to the same object, and get_type_info virtual function that
/// overrides Node::get_type_info and returns a reference to the same object that
/// get_type_info_static gives.
///
/// Use this macro as a public part of the class definition:
///
///     class MyClass
///     {
///         public:
///             OPENVINO_RTTI("MyClass", "my_version");
///
///             ...
///     };
///
///     class MyClass2: public MyClass
///     {
///         public:
///             OPENVINO_RTTI("MyClass2", "my_version2", MyClass);
///
///             ...
///     };
///
/// \param TYPE_NAME a string literal of type const char* that names your class in type
/// identification namespace;
///        It is your choice how to name it, but it should be unique among all
///        OPENVINO_RTTI_DECLARATION-enabled classes that can be
///        used in conjunction with each other in one transformation flow.
/// \param VERSION_NAME is an name of operation version to distinguish different versions of
///        operations that shares the same TYPE_NAME
/// \param PARENT_CLASS is an optional direct or indirect parent class for this class; define
///        it only in case if there is a need to capture any operation from some group of operations
///        that all derived from some common base class. Don't use Node as a parent, it is a base
///        class
///        for all operations and doesn't provide ability to define some perfect subset of
///        operations. PARENT_CLASS should define RTTI with OPENVINO_RTTI_{DECLARATION/DEFINITION}
///        macros.
///
/// OPENVINO_RTTI(name)
/// OPENVINO_RTTI(name, version_id)
/// OPENVINO_RTTI(name, version_id, parent)

#define OPENVINO_RTTI(...)                                                                              \
    _OPENVINO_RTTI_EXPAND(_OPENVINO_RTTI_DEFINITION_SELECTOR_3(__VA_ARGS__,                             \
                                                               _OPENVINO_RTTI_WITH_TYPE_VERSION_PARENT, \
                                                               _OPENVINO_RTTI_WITH_TYPE_VERSION,        \
                                                               _OPENVINO_RTTI_WITH_TYPE)(__VA_ARGS__))

#define _OPENVINO_RTTI_BASE_WITH_TYPE(TYPE_NAME) _OPENVINO_RTTI_BASE_WITH_TYPE_VERSION(TYPE_NAME, "util")

#define _OPENVINO_RTTI_BASE_WITH_TYPE_VERSION(TYPE_NAME, VERSION_NAME)                    \
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() { \
        static ::ov::DiscreteTypeInfo type_info_static{TYPE_NAME, VERSION_NAME};          \
        type_info_static.hash();                                                          \
        return type_info_static;                                                          \
    }                                                                                     \
    virtual const ::ov::DiscreteTypeInfo& get_type_info() const { return get_type_info_static(); }

/// Helper macro for base (without rtti parrent) class that provides RTTI block definition.
/// It's a two parameter version of OPENVINO_RTTI macro, without PARENT_CLASS.
#define OPENVINO_RTTI_BASE(...)                                                                       \
    _OPENVINO_RTTI_EXPAND(_OPENVINO_RTTI_DEFINITION_SELECTOR_2(__VA_ARGS__,                           \
                                                               _OPENVINO_RTTI_BASE_WITH_TYPE_VERSION, \
                                                               _OPENVINO_RTTI_BASE_WITH_TYPE)(__VA_ARGS__))

#if defined(__GNUC__)
#    define OPENVINO_DO_PRAGMA(x) _Pragma(#x)
#elif defined(_MSC_VER)
#    define OPENVINO_DO_PRAGMA(x) __pragma(x)
#else
#    define OPENVINO_DO_PRAGMA(x)
#endif

#if defined(__clang__)
#    if defined(__has_warning) && __has_warning("-Wsuggest-override")
#        define OPENVINO_SUPPRESS_SUGGEST_OVERRIDE_START                      \
            OPENVINO_DO_PRAGMA(clang diagnostic push)                         \
            OPENVINO_DO_PRAGMA(clang diagnostic ignored "-Wsuggest-override") \
            OPENVINO_DO_PRAGMA(clang diagnostic ignored "-Winconsistent-missing-override")
#    else
#        define OPENVINO_SUPPRESS_SUGGEST_OVERRIDE_START \
            OPENVINO_DO_PRAGMA(clang diagnostic push)    \
            OPENVINO_DO_PRAGMA(clang diagnostic ignored "-Winconsistent-missing-override")
#    endif  // defined(__has_warning) && __has_warning("-Wsuggest-override")
#    define OPENVINO_SUPPRESS_SUGGEST_OVERRIDE_END OPENVINO_DO_PRAGMA(clang diagnostic pop)
#elif (defined(__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ > 405))
#    define OPENVINO_SUPPRESS_SUGGEST_OVERRIDE_START \
        OPENVINO_DO_PRAGMA(GCC diagnostic push)      \
        OPENVINO_DO_PRAGMA(GCC diagnostic ignored "-Wsuggest-override")
#    define OPENVINO_SUPPRESS_SUGGEST_OVERRIDE_END OPENVINO_DO_PRAGMA(GCC diagnostic pop)
#elif defined(_MSC_VER)
#    define OPENVINO_SUPPRESS_SUGGEST_OVERRIDE_START \
        OPENVINO_DO_PRAGMA(warning(push))            \
        OPENVINO_DO_PRAGMA(warning(disable : 4373))
#    define OPENVINO_SUPPRESS_SUGGEST_OVERRIDE_END OPENVINO_DO_PRAGMA(warning(pop))
#else
#    define OPENVINO_SUPPRESS_SUGGEST_OVERRIDE_START
#    define OPENVINO_SUPPRESS_SUGGEST_OVERRIDE_END
#endif
