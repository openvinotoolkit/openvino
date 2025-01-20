// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"

#include "openvino/util/file_util.hpp"

ov::Exception::Exception(const std::string& what_arg) : std::runtime_error(what_arg) {}

void ov::Exception::create(const char* file, int line, const std::string& explanation) {
    throw ov::Exception(make_what(file, line, nullptr, default_msg, explanation));
}

std::string ov::Exception::make_what(const char* file,
                                     int line,
                                     const char* check_string,
                                     const std::string& context_info,
                                     const std::string& explanation) {
    std::stringstream ss;
    if (check_string) {
        ss << "Check '" << check_string << "' failed at " << util::trim_file_name(file) << ":" << line;
    } else {
        ss << "Exception from " << util::trim_file_name(file) << ":" << line;
    }
    if (!context_info.empty()) {
        ss << ":" << std::endl << context_info;
    }
    if (!explanation.empty()) {
        ss << ":" << std::endl << explanation;
    }
    ss << std::endl;
    return ss.str();
}

ov::Exception::~Exception() = default;

const std::string ov::Exception::default_msg{};

void ov::AssertFailure::create(const char* file,
                               int line,
                               const char* check_string,
                               const std::string& context_info,
                               const std::string& explanation) {
    throw ov::AssertFailure(make_what(file, line, check_string, context_info, explanation));
}

ov::AssertFailure::~AssertFailure() = default;

void ov::NotImplemented::create(const char* file, int line, const std::string& explanation) {
    throw ov::NotImplemented(make_what(file, line, nullptr, default_msg, explanation));
}

ov::NotImplemented::~NotImplemented() = default;

const std::string ov::NotImplemented::default_msg{"Not Implemented"};
