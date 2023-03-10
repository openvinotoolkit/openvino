// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/exception.hpp"

namespace ov {
namespace frontend {

void throw_general_failure(const CheckLocInfo& check_loc_info,
                           const std::string& context_info,
                           const std::string& explanation) {
    throw ov::frontend::GeneralFailure(check_loc_info, context_info, explanation);
}

void throw_initialization_failure(const CheckLocInfo& check_loc_info,
                                  const std::string& context_info,
                                  const std::string& explanation) {
    throw ov::frontend::InitializationFailure(check_loc_info, context_info, explanation);
}

void throw_op_conversion_failure(const CheckLocInfo& check_loc_info,
                                 const std::string& context_info,
                                 const std::string& explanation) {
    throw ov::frontend::OpConversionFailure(check_loc_info, context_info, explanation);
}

void throw_op_validation_failure(const CheckLocInfo& check_loc_info,
                                 const std::string& context_info,
                                 const std::string& explanation) {
    throw ov::frontend::OpValidationFailure(check_loc_info, context_info, explanation);
}

void throw_not_implemented(const CheckLocInfo& check_loc_info,
                           const std::string& context_info,
                           const std::string& explanation) {
    throw ov::frontend::NotImplementedFailure(check_loc_info, context_info, explanation);
}

}  // namespace frontend
}  // namespace ov
