// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "openvino/frontend/visibility.hpp"

#define RETHROW_FRONTEND_EXCEPTION(Type) \
    catch (const Type& ex) {             \
        throw Type(ex);                  \
    }

#define FRONTEND_CALL_STATEMENT(MESSAGE, ...)                       \
    try {                                                           \
        __VA_ARGS__;                                                \
    }                                                               \
    RETHROW_FRONTEND_EXCEPTION(ov::frontend::GeneralFailure)        \
    RETHROW_FRONTEND_EXCEPTION(ov::frontend::OpValidationFailure)   \
    RETHROW_FRONTEND_EXCEPTION(ov::frontend::InitializationFailure) \
    RETHROW_FRONTEND_EXCEPTION(ov::frontend::OpConversionFailure)   \
    RETHROW_FRONTEND_EXCEPTION(ov::frontend::NotImplementedFailure) \
    RETHROW_FRONTEND_EXCEPTION(ov::AssertFailure)                   \
    RETHROW_FRONTEND_EXCEPTION(ov::Exception)                       \
    catch (...) {                                                   \
        OPENVINO_ASSERT(false, (MESSAGE));                          \
    }

#define FRONTEND_RETURN_STATEMENT(MESSAGE, FUNCTION)                \
    try {                                                           \
        return FUNCTION;                                            \
    }                                                               \
    RETHROW_FRONTEND_EXCEPTION(ov::frontend::GeneralFailure)        \
    RETHROW_FRONTEND_EXCEPTION(ov::frontend::OpValidationFailure)   \
    RETHROW_FRONTEND_EXCEPTION(ov::frontend::InitializationFailure) \
    RETHROW_FRONTEND_EXCEPTION(ov::frontend::OpConversionFailure)   \
    RETHROW_FRONTEND_EXCEPTION(ov::frontend::NotImplementedFailure) \
    RETHROW_FRONTEND_EXCEPTION(ov::AssertFailure)                   \
    RETHROW_FRONTEND_EXCEPTION(ov::Exception)                       \
    catch (...) {                                                   \
        OPENVINO_ASSERT(false, (MESSAGE));                          \
    }

namespace ov {
namespace frontend {
std::string get_frontend_library_path();
}  // namespace frontend
}  // namespace ov
