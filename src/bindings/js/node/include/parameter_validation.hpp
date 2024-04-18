// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

std::string get_type_name(napi_valuetype type) {
    switch (type) {
    case napi_undefined:
        return "Undefined";

    case napi_null:
        return "Null";

    case napi_boolean:
        return "Boolean";

    case napi_number:
        return "Number";

    case napi_string:
        return "String";

    case napi_symbol:
        return "Symbol";

    case napi_object:
        return "Object";

    case napi_function:
        return "Function";

    case napi_external:
        return "External";

    case napi_bigint:
        return "BigInt";

    default:
        return "Unknown";
    }
}

struct NapiArgValidator {
    typedef std::function<void(const std::string& key, const Napi::Value&)> ValidatorType;
    std::vector<ValidatorType> attributes_validators;

    bool validate(const Napi::CallbackInfo& info, std::vector<std::string>& error_messages) {
        std::string validation_errors;

        if (info.Length() != attributes_validators.size()) return false;

        size_t index = 0;

        for (const auto& validator : attributes_validators) {
            try {
                validator(std::to_string(index), info[index]);
            } catch (std::runtime_error& err) {
                validation_errors.append("\t" + std::string(err.what()) + "\n");
            }

            index++;
        }

        error_messages.push_back(validation_errors);

        return validation_errors.empty();
    }

    void add_arg(ValidatorType validator) {
        attributes_validators.push_back(validator);
    }

    NapiArgValidator& add_boolean_arg() {
        attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
            check_type(napi_boolean, key, value);
        });

        return *this;
    }

    NapiArgValidator& add_number_arg() {
        attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
            check_type(napi_number, key, value);
        });

        return *this;
    }

    NapiArgValidator& add_string_arg() {
        attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
            check_type(napi_string, key, value);
        });

        return *this;
    }

    NapiArgValidator& add_symbol_arg() {
        attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
            check_type(napi_symbol, key, value);
        });

        return *this;
    }

    NapiArgValidator& add_object_arg() {
        attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
            check_type(napi_object, key, value);
        });

        return *this;
    }

    NapiArgValidator& add_function_arg() {
        attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
            check_type(napi_function, key, value);
        });

        return *this;
    }

    NapiArgValidator& add_bigint_arg() {
        attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
            check_type(napi_bigint, key, value);
        });

        return *this;
    }

    NapiArgValidator& add_array_arg() {
        attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
            OPENVINO_ASSERT(value.IsArray(), create_error_message(key, "Array", value.Type()));
        });

        return *this;
    }

    static void check_type(const napi_valuetype expected_type, const std::string key, const Napi::Value& value) {
        napi_valuetype real_type = value.Type();

        OPENVINO_ASSERT(real_type == expected_type, create_error_message(key, expected_type, real_type));
    }

    static const std::string create_error_message(std::string key,
                                               napi_valuetype expected_type,
                                               napi_valuetype real_type) {
        std::string expected_type_str = get_type_name(expected_type);
        std::string real_type_str = get_type_name(real_type);

        return create_error_message(key, expected_type_str, real_type_str);
    }

    static const std::string create_error_message(const std::string& key, const std::string& expected, const napi_valuetype& real_type) {
        std::string real_type_str = get_type_name(real_type);

        return create_error_message(key, expected, real_type_str);
    }

    static std::string create_error_message(std::string key, std::string expected, std::string real) {
        return "Argument #" + key + " has type '" + real + "', expected '" + expected + "'";
    }
};
