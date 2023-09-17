// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/crash_handler.hpp"

#include <limits.h>
#include <signal.h>

#include "functional_test_utils/summary/api_summary.hpp"
#include "functional_test_utils/summary/op_summary.hpp"

namespace ov {
namespace test {
namespace utils {

#if defined(__APPLE__)
typedef sig_t sighandler;
#elif defined(_WIN32)
#    ifdef __GNUC__
typedef __p_sig_fn_t sighandler;
#    else
typedef _crt_signal_t sighandler;
#    endif
#else
typedef sighandler_t sighandler;
#endif

// enviroment to restore in case of crash
jmp_buf env;
unsigned int CrashHandler::MAX_TEST_WORK_TIME = UINT_MAX;
bool CrashHandler::IGNORE_CRASH = false;

CrashHandler::CrashHandler(CONFORMANCE_TYPE type) {
    // setup default value for timeout in 15 minutes
    if (MAX_TEST_WORK_TIME == UINT_MAX) {
        MAX_TEST_WORK_TIME = 900;
    }

    sighandler crashHandler = [](int errCode) {
        std::cerr << "Unexpected application crash with code: " << errCode << std::endl;

        // reset custom signal handler to avoid infinit loop
        // if for some reasons sigsetjmp will not be available
        signal(SIGABRT, SIG_DFL);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGILL, SIG_DFL);
#ifndef _WIN32
        signal(SIGBUS, SIG_DFL);
        signal(SIGFPE, SIG_DFL);
        signal(SIGALRM, SIG_DFL);
#endif

        if (!CrashHandler::IGNORE_CRASH) {
            auto& s = ov::test::utils::OpSummary::getInstance();
            s.saveReport();
            std::abort();
        }

#ifdef _WIN32
        longjmp(env, JMP_STATUS::anyError);
#else
        // reset timeout
        alarm(0);

        if (errCode == SIGALRM) {
            std::cerr << "Test finished by timeout" << std::endl;
            siglongjmp(env, JMP_STATUS::alarmErr);
        } else {
            siglongjmp(env, JMP_STATUS::anyError);
        }
#endif
    };

    if (type == CONFORMANCE_TYPE::api) {
        crashHandler = [](int errCode) {
            std::cerr << "Unexpected application crash with code: " << errCode << ". Program will aborted."
                      << std::endl;

            // reset custom signal handler to avoid infinit loop
            // if for some reasons sigsetjmp will not be available
            signal(SIGABRT, SIG_DFL);
            signal(SIGSEGV, SIG_DFL);
            signal(SIGILL, SIG_DFL);
#ifndef _WIN32
            signal(SIGBUS, SIG_DFL);
            signal(SIGFPE, SIG_DFL);
            signal(SIGALRM, SIG_DFL);
#endif
            auto& s = ov::test::utils::ApiSummary::getInstance();
            s.saveReport();
            std::abort();
        };
    }

    // setup custom handler for signals
    signal(SIGABRT, crashHandler);
    signal(SIGSEGV, crashHandler);
    signal(SIGILL, crashHandler);
#ifndef _WIN32
    signal(SIGFPE, crashHandler);
    signal(SIGBUS, crashHandler);
    signal(SIGALRM, crashHandler);
#endif
}

CrashHandler::~CrashHandler() {
    // reset custom signal handler to avoid infinit loop
    signal(SIGABRT, SIG_DFL);
    signal(SIGSEGV, SIG_DFL);
    signal(SIGILL, SIG_DFL);
#ifndef _WIN32
    signal(SIGFPE, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGALRM, SIG_DFL);
#endif

    // reset timeout
#ifndef _WIN32
    alarm(0);
#endif
}

void CrashHandler::StartTimer() {
#ifndef _WIN32
    alarm(MAX_TEST_WORK_TIME);
#endif
}

void CrashHandler::SetUpTimeout(unsigned int timeout) {
    MAX_TEST_WORK_TIME = timeout;
}

void CrashHandler::SetUpPipelineAfterCrash(bool ignore_crash) {
    IGNORE_CRASH = ignore_crash;
}

}  // namespace utils
}  // namespace test
}  // namespace ov
