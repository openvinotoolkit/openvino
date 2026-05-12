// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>

#include "openvino/util/file_util.hpp"

namespace {
// [SD3-DBG] Unconditional dump of every OV exception text right before it is thrown.
// This is intentionally noisy and exists only to diagnose the stable-diffusion-v3
// GPU abort on Arc B390 where the closest-exported-symbol heuristic in the c10
// abort handler points at an unrelated symbol (isQuantizedStatic) and the actual
// exception message is otherwise lost when the unhandled C++ exception terminates
// the process via std::terminate.
inline void sd3_dbg_dump_throw(const char* kind, const std::string& what_text) {
    std::cout << "[SD3-DBG] OV THROW kind=" << kind << " what=" << what_text << std::flush;
    std::cerr << "[SD3-DBG] OV THROW kind=" << kind << " what=" << what_text << std::flush;
}

// [SD3-DBG] Global std::set_terminate backstop. Catches ANY uncaught exception
// (OV, std, raw cldnn throws, noexcept violations) and prints what() before the
// process aborts. This wins the race against torch's c10::TerminateHandler that
// otherwise eats the exception message and only prints a misleading nearest-symbol
// stack walk.
struct Sd3DbgTerminateInstaller {
    Sd3DbgTerminateInstaller() {
        m_prev = std::set_terminate(&Sd3DbgTerminateInstaller::handler);
        std::cout << "[SD3-DBG] std::set_terminate handler installed (prev=" << reinterpret_cast<void*>(m_prev) << ")" << std::endl;
    }
    static std::terminate_handler m_prev;
    static void handler() {
        try {
            auto eptr = std::current_exception();
            if (eptr) {
                try {
                    std::rethrow_exception(eptr);
                } catch (const std::exception& e) {
                    std::cout << "[SD3-DBG] TERMINATE uncaught std::exception type=" << typeid(e).name()
                              << " what=" << e.what() << std::endl;
                    std::cerr << "[SD3-DBG] TERMINATE uncaught std::exception type=" << typeid(e).name()
                              << " what=" << e.what() << std::endl;
                } catch (...) {
                    std::cout << "[SD3-DBG] TERMINATE uncaught non-std exception" << std::endl;
                    std::cerr << "[SD3-DBG] TERMINATE uncaught non-std exception" << std::endl;
                }
            } else {
                std::cout << "[SD3-DBG] TERMINATE called with no current_exception (direct std::terminate or noexcept violation)" << std::endl;
                std::cerr << "[SD3-DBG] TERMINATE called with no current_exception (direct std::terminate or noexcept violation)" << std::endl;
            }
        } catch (...) {
            // never let the handler itself throw
        }
        std::cout.flush();
        std::cerr.flush();
        // Chain to previous handler if any (likely torch's c10 handler) so behavior is unchanged.
        if (m_prev) {
            m_prev();
        }
        std::abort();
    }
};
std::terminate_handler Sd3DbgTerminateInstaller::m_prev = nullptr;
// Static initializer: installs as early as openvino.dll is loaded.
Sd3DbgTerminateInstaller g_sd3_dbg_terminate_installer;
}  // namespace

ov::Exception::Exception(const std::string& what_arg) : std::runtime_error(what_arg) {}

void ov::Exception::create(const char* file, int line, const std::string& explanation) {
    auto what_text = make_what(file, line, nullptr, default_msg, explanation);
    sd3_dbg_dump_throw("Exception", what_text);
    throw ov::Exception(what_text);
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
    auto what_text = make_what(file, line, check_string, context_info, explanation);
    sd3_dbg_dump_throw("AssertFailure", what_text);
    throw ov::AssertFailure(what_text);
}

ov::AssertFailure::~AssertFailure() = default;

void ov::NotImplemented::create(const char* file, int line, const std::string& explanation) {
    auto what_text = make_what(file, line, nullptr, default_msg, explanation);
    sd3_dbg_dump_throw("NotImplemented", what_text);
    throw ov::NotImplemented(what_text);
}

ov::NotImplemented::~NotImplemented() = default;

const std::string ov::NotImplemented::default_msg{"Not Implemented"};

namespace ov {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
std::string stringify(const std::filesystem::path& arg) {
    return std::string("\"") + ov::util::path_to_string(arg) + '"';
}
#endif
}  // namespace ov
