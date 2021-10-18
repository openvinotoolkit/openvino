// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef MULTIDEVICEPLUGIN_THREADUTILS_H
#define MULTIDEVICEPLUGIN_THREADUTILS_H

#include <cstdint>
#include <string>

#ifdef WIN32
#include <windows.h>
#include <winsock.h>
#include <intrin.h>
namespace ThreadUtils {
DWORD getThreadId();
} // namespace ThreadUtils
#else
#include <sys/types.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace ThreadUtils {
long getThreadId();
} // namespace ThreadUtils
#endif

namespace ThreadUtils {
uint64_t getRdtsc();
std::string getName();
std::string getHostname();

void setName(const char* name);
void setName(const std::string& name);
void getName(char* name, size_t size);
void saveThreadInfo(const std::string& threadName, const std::string& saveFile = "threads.info");
} // namespace ThreadUtils

#endif //MULTIDEVICEPLUGIN_THREADUTILS_H
