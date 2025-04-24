// Copyright (C) 2018-2025 Intel Corporation
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
    catch (const std::exception& e) {                               \
        const auto message = std::string(MESSAGE "\n") + e.what();  \
        OPENVINO_ASSERT(false, message);                            \
    }                                                               \
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
    catch (const std::exception& e) {                               \
        const auto message = std::string(MESSAGE "\n") + e.what();  \
        OPENVINO_ASSERT(false, message);                            \
    }                                                               \
    catch (...) {                                                   \
        OPENVINO_ASSERT(false, (MESSAGE));                          \
    }
