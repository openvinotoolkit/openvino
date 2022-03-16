// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "crash_handler.hpp"
#include <limits.h>

#ifndef _WIN32
#include <sys/resource.h>
#endif

namespace CommonTestUtils {

// enviroment to restore in case of crash
jmp_buf env;
unsigned int CrashHandler::MAX_TEST_WORK_TIME = UINT_MAX;
unsigned long long CrashHandler::MAX_VIRTUAL_MEMORY = 0;

CrashHandler::CrashHandler() {
    // setup default value for timeout in 15 minutes
    if (MAX_TEST_WORK_TIME == UINT_MAX) {
        MAX_TEST_WORK_TIME = 900;
    }

    auto crashHandler = [](int errCode) {
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

void CrashHandler::SetUpVMLimit(unsigned long long memory_limit) {
    MAX_VIRTUAL_MEMORY = long(1024)*memory_limit;

    if (memory_limit == 0) {
        return;
    }

#ifndef _WIN32
    struct rlimit memlimit;
    memlimit.rlim_cur = MAX_VIRTUAL_MEMORY;
    memlimit.rlim_max = MAX_VIRTUAL_MEMORY;
    setrlimit(RLIMIT_AS, &memlimit);
#endif
}
}  // namespace CommonTestUtils