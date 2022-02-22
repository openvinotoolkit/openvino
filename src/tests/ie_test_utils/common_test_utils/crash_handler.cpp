// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "crash_handler.hpp"

namespace CommonTestUtils {

// enviroment to restore in case of crash
jmp_buf env;

CrashHandler::CrashHandler() {
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
#endif

        // goto sigsetjmp
#ifdef _WIN32
        longjmp(env, 1);
#else
        siglongjmp(env, 1);
#endif
    };

    // setup custom handler for signals
    signal(SIGABRT, crashHandler);
    signal(SIGSEGV, crashHandler);
    signal(SIGILL, crashHandler);
#ifndef _WIN32
    signal(SIGFPE, crashHandler);
    signal(SIGBUS, crashHandler);
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
#endif
}

}  // namespace CommonTestUtils