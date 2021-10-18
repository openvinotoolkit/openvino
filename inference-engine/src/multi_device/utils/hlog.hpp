// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef MULTIDEVICEPLUGIN_HLOG_H
#define MULTIDEVICEPLUGIN_HLOG_H

#include <cstdarg>

#include "log.hpp"

#define HLogger MultiDevicePlugin::Log::instance()

#define HLogPrint(isOn, isTraceCallStack, logLevel, level, tag, ...) \
    HLogger->doLog(isOn, isTraceCallStack, logLevel, level, __FILE__, __func__, __LINE__, tag, __VA_ARGS__)

#define HFrequent(isOn, tag, ...) HLogPrint(isOn, MultiDevicePlugin::LogLevel::FREQUENT, "FREQ", tag, __VA_ARGS__)
#define HProcess(isOn, tag, ...) HLogPrint(isOn, false, MultiDevicePlugin::LogLevel::PROCESS, "PROC", tag, __VA_ARGS__)
#define HDebug(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::DEBUG, "DEBUG", nullptr, __VA_ARGS__)
#define HInfo(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::INFO, "INFO", nullptr, __VA_ARGS__)
#define HWarn(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::WARN, "WARN", nullptr, __VA_ARGS__)
#define HError(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::ERROR, "ERROR", nullptr, __VA_ARGS__)
#define HFatal(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::FATAL, "FATAL", nullptr, __VA_ARGS__)

#define TraceCallStacks(...) HLogPrint(true, true, MultiDevicePlugin::LogLevel::DEBUG, "DEBUG", nullptr, __VA_ARGS__)
#define TraceCallStack() TraceCallStacks(" ")

namespace MultiDevicePlugin {
inline void setLogLevel(uint32_t logLevel) {
    HLogger->setLogLevel(logLevel);
}

inline void setLogName(std::string logName) {
    HLogger->setLogName(logName);
}

} // namespace MultiDevicePlugin

#endif //MULTIDEVICEPLUGIN_HLOG_H
