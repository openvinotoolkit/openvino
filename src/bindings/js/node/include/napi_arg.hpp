// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/runtime/core.hpp"

namespace NapiTypename {
const std::string UNDEFINED_STR = "Undefined";
const std::string NULL_STR = "Null";
const std::string BOOLEAN_STR = "Boolean";
const std::string NUMBER_STR = "Number";
const std::string STIRNG_STR = "String";
const std::string SYMBOL_STR = "Symbol";
const std::string OBJECT_STR = "Object";
const std::string FUNCTION_STR = "Function";
const std::string EXTERNAL_STR = "External";
const std::string BIGINT_STR = "BigInt";
const std::string UNKNOWN_STR = "Unknown";
}  // namespace NapiTypename

namespace NapiArg {
std::string join_array_to_str(std::vector<std::string> array, std::string separator);

const std::string& get_type_name(napi_valuetype type);

std::string create_error_message(const std::string& key, const std::string& expected, const std::string real);

std::string create_error_message(const std::string key, napi_valuetype expected_type, napi_valuetype real_type);

std::string create_error_message(const std::string& key,
                                       const std::string& expected,
                                       const napi_valuetype& real_type);

void check_type(const napi_valuetype expected_type, const std::string key, const Napi::Value& value);

struct Validator {
public:
    typedef std::function<void(const std::string& key, const Napi::Value&)> ValidatorType;

    const bool validate(const Napi::CallbackInfo& info, std::vector<std::string>& error_messages);

    Validator& add_arg(ValidatorType validator);

    Validator& add_boolean_arg();

    Validator& add_number_arg();

    Validator& add_string_arg();

    Validator& add_symbol_arg();

    Validator& add_object_arg();

    Validator& add_function_arg();

    Validator& add_bigint_arg();

    Validator& add_array_arg();

private:
    std::vector<ValidatorType> attributes_validators;
};
}  // namespace NapiArg
