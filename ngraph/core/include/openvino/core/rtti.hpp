// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type.hpp"

#define _OPENVINO_RTTI_EXPAND(X)                                      X
#define _OPENVINO_RTTI_DEFINITION_SELECTOR(_1, _2, _3, _4, NAME, ...) NAME

#define _OPENVINO_RTTI_DEF_WITH_TYPE(CLASS, TYPE_NAME) _OPENVINO_RTTI_DEF_WITH_TYPE_VERSION(CLASS, TYPE_NAME, "util")

#define _OPENVINO_RTTI_DEF_WITH_TYPE_VERSION(CLASS, TYPE_NAME, VERSION_NAME) \
    _OPENVINO_RTTI_DEF_WITH_TYPE_VERSION_PARENT(CLASS, TYPE_NAME, VERSION_NAME, ::ov::op::Op)

#define _OPENVINO_RTTI_DEF_WITH_TYPE_VERSION_PARENT(CLASS, TYPE_NAME, VERSION_NAME, PARENT_CLASS) \
    _OPENVINO_RTTI_DEF_WITH_TYPE_VERSIONS_PARENT(CLASS, TYPE_NAME, VERSION_NAME, PARENT_CLASS, 0)

#define _OPENVINO_RTTI_DEF_WITH_TYPE_VERSIONS_PARENT(CLASS, TYPE_NAME, VERSION_NAME, PARENT_CLASS, OLD_VERSION) \
    const ov::DiscreteTypeInfo CLASS::type_info{TYPE_NAME,                                                      \
                                                OLD_VERSION,                                                    \
                                                VERSION_NAME,                                                   \
                                                &PARENT_CLASS::get_type_info_static()};                         \
    const ov::DiscreteTypeInfo& CLASS::get_type_info() const {                                                  \
        return type_info;                                                                                       \
    }

#define _OPENVINO_RTTI_WITH_TYPE(TYPE_NAME) _OPENVINO_RTTI_WITH_TYPE_VERSION(TYPE_NAME, "util")

#define _OPENVINO_RTTI_WITH_TYPE_VERSION(TYPE_NAME, VERSION_NAME)                  \
    static const ::ov::DiscreteTypeInfo& get_type_info_static() {                  \
        static const ::ov::DiscreteTypeInfo type_info{TYPE_NAME, 0, VERSION_NAME}; \
        return type_info;                                                          \
    }                                                                              \
    const ::ov::DiscreteTypeInfo& get_type_info() const override {                 \
        return get_type_info_static();                                             \
    }

#define _OPENVINO_RTTI_WITH_TYPE_VERSION_PARENT(TYPE_NAME, VERSION_NAME, PARENT_CLASS) \
    _OPENVINO_RTTI_WITH_TYPE_VERSIONS_PARENT(TYPE_NAME, VERSION_NAME, PARENT_CLASS, 0)

#define _OPENVINO_RTTI_WITH_TYPE_VERSIONS_PARENT(TYPE_NAME, VERSION_NAME, PARENT_CLASS, OLD_VERSION) \
    static const ::ov::DiscreteTypeInfo& get_type_info_static() {                                    \
        static const ::ov::DiscreteTypeInfo type_info{TYPE_NAME,                                     \
                                                      OLD_VERSION,                                   \
                                                      VERSION_NAME,                                  \
                                                      &PARENT_CLASS::get_type_info_static()};        \
        return type_info;                                                                            \
    }                                                                                                \
    const ::ov::DiscreteTypeInfo& get_type_info() const override {                                   \
        return get_type_info_static();                                                               \
    }
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
///     class MyOp : public Node
///     {
///         public:
///             // Don't use Node as a parent for type_info, it doesn't have any value and
///             prohibited
///             OPENVINO_RTTI_DECLARATION;
///
///             ...
///     };
///
///     class MyInheritedOp : public MyOp
///     {
///         public:
///             OPENVINO_RTTI_DECLARATION;
///
///             ...
///     };
///
/// To complete type identification for a class, use OPENVINO_RTTI_DEFINITION.
///
#define OPENVINO_RTTI_DECLARATION                \
    static const ov::DiscreteTypeInfo type_info; \
    const ov::DiscreteTypeInfo& get_type_info() const override;

/// Complementary to OPENVINO_RTTI_DECLARATION, this helper macro _defines_ items _declared_ by
/// OPENVINO_RTTI_DECLARATION.
/// Should be used outside the class definition scope in place where ODR is ensured.
///
/// \param CLASS is a C++ name of the class where corresponding OPENVINO_RTTI_DECLARATION was applied.
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
/// \param _VERSION_INDEX is an unsigned integer index to distinguish different versions of
///        operations that shares the same TYPE_NAME (for backward compatibility)
///
/// Examples (see corresponding declarations in OPENVINO_RTTI_DECLARATION description):
///
///     OPENVINO_RTTI_DEFINITION(MyOp,"MyOp", 1);
///     OPENVINO_RTTI_DEFINITION(MyInheritedOp, "MyInheritedOp", 1, MyOp)
///
/// For convenience, TYPE_NAME and CLASS name are recommended to be the same.
///
/// OPENVINO_RTTI(name)
/// OPENVINO_RTTI(name, version_id)
/// OPENVINO_RTTI(name, version_id, parent)
/// OPENVINO_RTTI(name, version_id, parent, old_version)
#define OPENVINO_RTTI_DEFINITION(CLASS, ...)                                                               \
    _OPENVINO_RTTI_EXPAND(_OPENVINO_RTTI_DEFINITION_SELECTOR(__VA_ARGS__,                                  \
                                                             _OPENVINO_RTTI_DEF_WITH_TYPE_VERSIONS_PARENT, \
                                                             _OPENVINO_RTTI_DEF_WITH_TYPE_VERSION_PARENT,  \
                                                             _OPENVINO_RTTI_DEF_WITH_TYPE_VERSION,         \
                                                             _OPENVINO_RTTI_DEF_WITH_TYPE)(CLASS, __VA_ARGS__))

#define OPENVINO_RTTI(...)                                                                             \
    _OPENVINO_RTTI_EXPAND(_OPENVINO_RTTI_DEFINITION_SELECTOR(__VA_ARGS__,                              \
                                                             _OPENVINO_RTTI_WITH_TYPE_VERSIONS_PARENT, \
                                                             _OPENVINO_RTTI_WITH_TYPE_VERSION_PARENT,  \
                                                             _OPENVINO_RTTI_WITH_TYPE_VERSION,         \
                                                             _OPENVINO_RTTI_WITH_TYPE)(__VA_ARGS__))

/// Note: Please don't use this macros for new operations
#define BWDCMP_RTTI_DECLARATION                                                                    \
    OPENVINO_DEPRECATED("This member was deprecate. Please use ::get_type_info_static() instead.") \
    static const ov::DiscreteTypeInfo type_info
#define BWDCMP_RTTI_DEFINITION(CLASS)                                            \
    OPENVINO_SUPPRESS_DEPRECATED_START                                           \
    const ov::DiscreteTypeInfo CLASS::type_info = CLASS::get_type_info_static(); \
    OPENVINO_SUPPRESS_DEPRECATED_END
