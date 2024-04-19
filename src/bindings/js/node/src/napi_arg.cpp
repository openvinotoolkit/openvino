// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/napi_arg.hpp"

namespace NapiArg {
std::string join_array_to_str(std::vector<std::string> array, std::string separator) {
    if (array.empty())
        return "";

    std::string result = array[0];
    for (size_t i = 1; i < array.size(); ++i) {
        result += separator + array[i];
    }

    return result;
}

const std::string& get_type_name(napi_valuetype type) {
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

std::string create_error_message(const std::string& key, const std::string& expected, const std::string real) {
    return "Argument #" + key + " has type '" + real + "', expected '" + expected + "'";
}

std::string create_error_message(const std::string key, napi_valuetype expected_type, napi_valuetype real_type) {
    std::string expected_type_str = get_type_name(expected_type);
    std::string real_type_str = get_type_name(real_type);

    return create_error_message(key, expected_type_str, real_type_str);
}

std::string create_error_message(const std::string& key, const std::string& expected, const napi_valuetype& real_type) {
    std::string real_type_str = get_type_name(real_type);

    return create_error_message(key, expected, real_type_str);
}

void check_type(const napi_valuetype expected_type, const std::string key, const Napi::Value& value) {
    napi_valuetype real_type = value.Type();

    OPENVINO_ASSERT(real_type == expected_type, create_error_message(key, expected_type, real_type));
}

const bool Validator::validate(const Napi::CallbackInfo& info) {
    std::string validation_errors;

    if (info.Length() != attributes_validators.size())
        return false;

    size_t index = 0;

    for (const auto& validator : attributes_validators) {
        try {
            validator(std::to_string(index + 1), info[index]);
        } catch (std::runtime_error& err) {
            validation_errors.append("\t" + std::string(err.what()) + "\n");
        }

        index++;
    }

    bool no_errors = validation_errors.empty();

    OPENVINO_ASSERT(no_errors, validation_errors);

    return no_errors;
}

const bool Validator::validate(const Napi::CallbackInfo& info, std::vector<std::string>& error_messages) {
    std::string validation_errors;

    if (info.Length() != attributes_validators.size())
        return false;

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

Validator& Validator::add_arg(ValidatorType validator) {
    attributes_validators.push_back(validator);

    return *this;
}

Validator& Validator::add_boolean_arg() {
    attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
        check_type(napi_boolean, key, value);
    });

    return *this;
}

Validator& Validator::add_number_arg() {
    attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
        check_type(napi_number, key, value);
    });

    return *this;
}

Validator& Validator::add_string_arg() {
    attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
        check_type(napi_string, key, value);
    });

    return *this;
}

Validator& Validator::add_symbol_arg() {
    attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
        check_type(napi_symbol, key, value);
    });

    return *this;
}

Validator& Validator::add_object_arg() {
    attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
        check_type(napi_object, key, value);
    });

    return *this;
}

Validator& Validator::add_function_arg() {
    attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
        check_type(napi_function, key, value);
    });

    return *this;
}

Validator& Validator::add_bigint_arg() {
    attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
        check_type(napi_bigint, key, value);
    });

    return *this;
}

Validator& Validator::add_array_arg() {
    attributes_validators.push_back([](const std::string key, const Napi::Value& value) {
        OPENVINO_ASSERT(value.IsArray(), create_error_message(key, "Array", value.Type()));
    });

    return *this;
}
}  // namespace NapiArg
