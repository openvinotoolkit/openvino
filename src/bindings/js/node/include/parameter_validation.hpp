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

struct Signature {
    typedef std::function<void(const std::string key, const Napi::Value&)> ValidatorType;
    std::vector<ValidatorType> attributes_validators;

    void param(ValidatorType validator) {
        attributes_validators.push_back(validator);
    }

    static void boolean(const std::string key, const Napi::Value& value) {
        return check_type(napi_boolean, key, value);
    }

    static void number(const std::string key, const Napi::Value& value) {
        return check_type(napi_number, key, value);
    }

    static void string(const std::string key, const Napi::Value& value) {
        return check_type(napi_string, key, value);
    }

    static void symbol(const std::string key, const Napi::Value& value) {
        return check_type(napi_symbol, key, value);
    }

    static void object(const std::string key, const Napi::Value& value) {
        return check_type(napi_object, key, value);
    }

    static void function(const std::string key, const Napi::Value& value) {
        return check_type(napi_function, key, value);
    }

    static void bigint(const std::string key, const Napi::Value& value) {
        return check_type(napi_bigint, key, value);
    }

    static void array(const std::string key, const Napi::Value& value) {
        OPENVINO_ASSERT(value.IsArray(), get_error_message(key, "Array", value.Type()));
    }

    static void check_type(const napi_valuetype expected_type, const std::string key, const Napi::Value& value) {
        napi_valuetype real_type = value.Type();

        OPENVINO_ASSERT(real_type == expected_type, get_error_message(key, expected_type, real_type));
    }

    static const std::string get_error_message(std::string key,
                                               napi_valuetype expected_type,
                                               napi_valuetype real_type) {
        std::string expected_type_str = get_type_name(expected_type);
        std::string real_type_str = get_type_name(real_type);

        return get_error_message(key, expected_type_str, real_type_str);
    }

    static const std::string get_error_message(std::string key, std::string expected, napi_valuetype real_type) {
        std::string real_type_str = get_type_name(real_type);

        return get_error_message(key, expected, real_type_str);
    }

    static std::string get_error_message(std::string key, std::string expected, std::string real) {
        return "Argument #" + key + " has type '" + real + "', expected '" + expected + "'";
    }
};

std::pair<bool, const std::string> validate_args(const Napi::CallbackInfo& info,
                                                 const std::function<void(Signature&)>& builder) {
    Signature s = Signature();
    std::string validation_errors;

    builder(s);

    if (info.Length() != s.attributes_validators.size())
        return std::make_pair(false, validation_errors);

    size_t index = 0;

    for (const auto& validator : s.attributes_validators) {
        try {
            validator(std::to_string(index), info[index]);
        } catch (std::runtime_error& err) {
            validation_errors.append("\t" + std::string(err.what()) + "\n");
        }

        index++;
    }

    bool is_errors_empty = validation_errors.empty();

    return std::make_pair(is_errors_empty, validation_errors);
}

std::function<bool(const Napi::CallbackInfo& info)> create_signature(const std::function<void(Signature&)>& builder,
                                                                     std::vector<std::string>& error_messages) {
    return [builder, &error_messages](const Napi::CallbackInfo& info) {
        auto result = validate_args(info, builder);

        if (!result.first)
            error_messages.push_back(result.second);

        return result.first;
    };
}
