// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iomanip>

#include "thread_utils.hpp"
#include <mutex>

namespace ThreadUtils {
void setName(const std::string& name) {
    setName(name.c_str());
}

std::string getName() {
    char threadName[32] = { '\0' };
    getName(threadName, 32);
    return threadName;
}



void saveThreadInfo(const std::string& threadName, const std::string& saveFile) {
    static std::mutex mutex;
    static uint32_t threadCount = 0;

    std::lock_guard<std::mutex> autoLock(mutex);

    std::ofstream stream;
    if (!threadCount) {
        stream.open(saveFile, std::ios::out | std::ios::trunc);
    } else {
        stream.open(saveFile, std::ios::out | std::ios::app);
    }

    stream << "[" << std::setw(2) << std::setfill('0') << threadCount++ << "]"
           << " threadId=" << getThreadId() << " threadName=" << threadName << std::endl;
}

#ifdef WIN32
DWORD getThreadId() {
    return GetCurrentThreadId();
}

static void setThreadName(DWORD dwThreadID, const char* threadName) {
    HANDLE handle = OpenThread(THREAD_SET_LIMITED_INFORMATION, true, dwThreadID);
    auto len = MultiByteToWideChar(CP_ACP, 0, threadName, static_cast<int>(strlen(threadName)), NULL, 0);
    PWSTR threadDesc = new WCHAR[len + 1];
    MultiByteToWideChar(CP_ACP, 0, threadName, static_cast<int>(strlen(threadName)), threadDesc, len);
    threadDesc[len] = '\0';
    SetThreadDescription(handle, threadDesc);
    delete[] threadDesc;
    CloseHandle(handle);
}

void setName(const char* pname) {
    setThreadName(GetCurrentThreadId(), pname);
}

void getName(char* name, size_t size) {
    HANDLE handle = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, true, GetCurrentThreadId());
    PWSTR threadDesc;
    if (SUCCEEDED(GetThreadDescription(handle, &threadDesc))) {
        auto len = WideCharToMultiByte(CP_ACP, 0, threadDesc, static_cast<int>(wcslen(threadDesc)), NULL, 0, NULL, NULL);
        if (len < size) {
            WideCharToMultiByte(CP_ACP, 0, threadDesc, static_cast<int>(wcslen(threadDesc)), name, len, NULL, NULL);
            name[len] = '\0';
        }
        LocalFree(threadDesc);
    }
    CloseHandle(handle);
}
#else
long getThreadId() {
    return syscall(SYS_gettid);
}

void setName(const char* pname) {
    prctl(PR_SET_NAME, pname);
}

void getName(char* name, size_t size) {
    if (name && size >= 16) {
        prctl(PR_GET_NAME, name);
    }
}

#endif
} // namespace ThreadUtils


