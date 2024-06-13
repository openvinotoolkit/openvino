// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/type_validation.hpp"

namespace ov {
namespace js {
namespace NapiArg {
const char* get_type_name(napi_valuetype type) {
    switch (type) {
    case napi_undefined:
        return NapiTypename::UNDEFINED_STR;

    case napi_null:
        return NapiTypename::NULL_STR;

    case napi_boolean:
        return NapiTypename::BOOLEAN_STR;

    case napi_number:
        return NapiTypename::NUMBER_STR;

    case napi_string:
        return NapiTypename::STIRNG_STR;

    case napi_symbol:
        return NapiTypename::SYMBOL_STR;

    case napi_object:
        return NapiTypename::OBJECT_STR;

    case napi_function:
        return NapiTypename::FUNCTION_STR;

    case napi_external:
        return NapiTypename::EXTERNAL_STR;

    case napi_bigint:
        return NapiTypename::BIGINT_STR;

    default:
        return NapiTypename::UNKNOWN_STR;
    }
}
}  // namespace NapiArg

std::string get_current_signature(const Napi::CallbackInfo& info) {
    std::vector<std::string> signature_attributes;
    size_t attrs_length = info.Length();

    for (size_t i = 0; i < attrs_length; ++i) {
        signature_attributes.push_back(NapiArg::get_type_name(info[i].Type()));
    }

    return std::string("(" + ov::util::join(signature_attributes) + ")");
};

template <>
const char* get_attr_type<Napi::String>() {
    return NapiArg::get_type_name(napi_string);
}

template <>
const char* get_attr_type<Napi::Object>() {
    return NapiArg::get_type_name(napi_object);
}

template <>
const char* get_attr_type<Napi::Buffer<uint8_t>>() {
    return BindingTypename::BUFFER;
}

template <>
const char* get_attr_type<int>() {
    return BindingTypename::INT;
}

template <>
const char* get_attr_type<ModelWrap>() {
    return BindingTypename::MODEL;
}

template <>
const char* get_attr_type<TensorWrap>() {
    return BindingTypename::TENSOR;
}

template <>
bool validate_value<Napi::String>(const Napi::Env& env, const Napi::Value& value) {
    return napi_string == value.Type();
}

template <>
bool validate_value<Napi::Object>(const Napi::Env& env, const Napi::Value& value) {
    return napi_object == value.Type();
}

template <>
bool validate_value<Napi::Buffer<uint8_t>>(const Napi::Env& env, const Napi::Value& value) {
    return value.IsBuffer();
}

template <>
bool validate_value<int>(const Napi::Env& env, const Napi::Value& value) {
    return value.IsNumber() && env.Global()
                                   .Get("Number")
                                   .ToObject()
                                   .Get("isInteger")
                                   .As<Napi::Function>()
                                   .Call({value.ToNumber()})
                                   .ToBoolean()
                                   .Value();
}

template <>
bool validate_value<ModelWrap>(const Napi::Env& env, const Napi::Value& value) {
    const auto& prototype = env.GetInstanceData<AddonData>()->model;

    return value.ToObject().InstanceOf(prototype.Value().As<Napi::Function>());
}

template <>
bool validate_value<TensorWrap>(const Napi::Env& env, const Napi::Value& value) {
    const auto& prototype = env.GetInstanceData<AddonData>()->tensor;

    return value.ToObject().InstanceOf(prototype.Value().As<Napi::Function>());
}

std::string get_parameters_error_msg(const Napi::CallbackInfo& info, std::vector<std::string>& checked_signatures) {
    return "Method 'compileModelSync' called with incorrect parameters.\nProvided signature: " +
           js::get_current_signature(info) + " \nAllowed signatures:\n- " + ov::util::join(checked_signatures, "\n- ");
}
}  // namespace js
}  // namespace ov
