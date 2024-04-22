// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/runtime/core.hpp"

namespace NapiArg {
const char* get_type_name(napi_valuetype type);

std::string create_error_message(const std::string& key, const char* expected, const char* real);

std::string create_error_message(const std::string& key, napi_valuetype expected_type, napi_valuetype real_type);

std::string create_error_message(const std::string& key, const char* expected, const napi_valuetype& real_type);

void check_type(const napi_valuetype expected_type, const std::string& key, const Napi::Value& value);

class Validator {
public:
    typedef std::function<void(const std::string& key, const Napi::Value&)> ValidatorType;

    const bool validate(const Napi::CallbackInfo& info, std::vector<std::string>& error_messages);
    const bool validate(const Napi::CallbackInfo& info);

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
