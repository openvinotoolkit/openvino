// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/exception.hpp"

void ov::frontend::GeneralFailure::create(const char* file,
                                          int line,
                                          const char* check_string,
                                          const std::string& context_info,
                                          const std::string& explanation) {
    throw ov::frontend::GeneralFailure(
        make_what(file, line, check_string, "FrontEnd API failed with GeneralFailure" + context_info, explanation));
}

void ov::frontend::InitializationFailure::create(const char* file,
                                                 int line,
                                                 const char* check_string,
                                                 const std::string& context_info,
                                                 const std::string& explanation) {
    throw ov::frontend::InitializationFailure(make_what(file,
                                                        line,
                                                        check_string,
                                                        "FrontEnd API failed with InitializationFailure" + context_info,
                                                        explanation));
}

void ov::frontend::OpValidationFailure::create(const char* file,
                                               int line,
                                               const char* check_string,
                                               const std::string& context_info,
                                               const std::string& explanation) {
    throw ov::frontend::OpValidationFailure(make_what(file,
                                                      line,
                                                      check_string,
                                                      "FrontEnd API failed with OpValidationFailure" + context_info,
                                                      explanation));
}

void ov::frontend::OpConversionFailure::create(const char* file,
                                               int line,
                                               const char* check_string,
                                               const std::string& context_info,
                                               const std::string& explanation) {
    throw ov::frontend::OpConversionFailure(make_what(file,
                                                      line,
                                                      check_string,
                                                      "FrontEnd API failed with OpConversionFailure" + context_info,
                                                      explanation));
}

void ov::frontend::NotImplementedFailure::create(const char* file,
                                                 int line,
                                                 const char* check_string,
                                                 const std::string& context_info,
                                                 const std::string& explanation) {
    throw ov::frontend::NotImplementedFailure(make_what(file,
                                                        line,
                                                        check_string,
                                                        "FrontEnd API failed with NotImplementedFailure" + context_info,
                                                        explanation));
}
