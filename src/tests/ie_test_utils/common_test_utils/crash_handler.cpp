// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/include/functional_test_utils/layer_test_utils/summary.hpp"

#include "crash_handler.hpp"
#include <limits.h>

namespace CommonTestUtils {

// enviroment to restore in case of crash
jmp_buf env;
unsigned int CrashHandler::MAX_TEST_WORK_TIME = UINT_MAX;

CrashHandler::CrashHandler(bool continueWorkAfterCrash) {
    // setup default value for timeout in 15 minutes
    if (MAX_TEST_WORK_TIME == UINT_MAX) {
        MAX_TEST_WORK_TIME = 900;
    }

    auto handleCrashAndExit = [](int errCode) {
        std::cerr << "Unexpected application crash with code: " << errCode << std::endl;
        auto& s = LayerTestsUtils::Summary::getInstance();
        s.saveReport();
#ifndef _WIN32
        alarm(0);
#endif
        exit(errCode);
    };

    auto handleCrashAndContinue = [](int errCode) {
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
        exit(errCode);

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

    if (continueWorkAfterCrash) {
    // setup custom handler for signals
        signal(SIGABRT, handleCrashAndContinue);
        signal(SIGSEGV, handleCrashAndContinue);
        signal(SIGILL, handleCrashAndContinue);
#ifndef _WIN32
        signal(SIGFPE, handleCrashAndContinue);
        signal(SIGBUS, handleCrashAndContinue);
        signal(SIGALRM, handleCrashAndContinue);
#endif
    } else {
        // setup custom handler for signals
        signal(SIGABRT, handleCrashAndExit);
        signal(SIGSEGV, handleCrashAndExit);
        signal(SIGILL, handleCrashAndExit);
#ifndef _WIN32
        signal(SIGFPE, handleCrashAndExit);
        signal(SIGBUS, handleCrashAndExit);
        signal(SIGALRM, handleCrashAndExit);
#endif
    }
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

}  // namespace CommonTestUtils