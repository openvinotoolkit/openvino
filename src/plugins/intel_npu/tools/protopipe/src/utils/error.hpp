//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <sstream>

namespace details {

[[noreturn]] inline void assert_abort(const char* str, const int line, const char* file, const char* func) {
    std::stringstream ss;
    ss << file << ":" << line << ": Assertion " << str << " in function " << func << " failed\n";
    std::cerr << ss.str() << std::flush;
    abort();
}

[[noreturn]] inline void throw_error(const char* str) {
    std::stringstream ss;
    ss << "An exception thrown! " << str << std::flush;
    throw std::logic_error(ss.str());
}

}  // namespace details

#define ASSERT(expr)                                                      \
    {                                                                     \
        if (!(expr))                                                      \
            ::details::assert_abort(#expr, __LINE__, __FILE__, __func__); \
    }

#define THROW_ERROR(msg)                          \
    {                                             \
        std::ostringstream os;                    \
        os << msg;                                \
        ::details::throw_error(os.str().c_str()); \
    }
