// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "openvino/frontend/visibility.hpp"

#define FRONTEND_CALL_STATEMENT(MESSAGE, ...) \
    try {                                     \
        __VA_ARGS__;                          \
    } catch (const ov::Exception& ex) {       \
        throw ov::Exception(ex.what());       \
    } catch (...) {                           \
        OPENVINO_ASSERT(false, (MESSAGE));    \
    }

#define FRONTEND_RETURN_STATEMENT(MESSAGE, FUNCTION) \
    try {                                            \
        return FUNCTION;                             \
    } catch (const ov::Exception& ex) {              \
        throw ov::Exception(ex.what());              \
    } catch (...) {                                  \
        OPENVINO_ASSERT(false, (MESSAGE));           \
    }

namespace ov {
namespace frontend {
std::string get_frontend_library_path();
}  // namespace frontend
}  // namespace ov
