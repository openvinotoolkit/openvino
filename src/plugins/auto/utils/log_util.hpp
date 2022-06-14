// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef MULTIDEVICEPLUGIN_HLOG_H
#define MULTIDEVICEPLUGIN_HLOG_H

#include <cstdarg>

#include "log.hpp"
#include <ie_plugin_config.hpp>

#ifdef  MULTIUNITTEST
#include "plugin/mock_log_utils.hpp"
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#define HLogger MockMultiDevice::MockLog::GetInstance()
#else
#define MOCKTESTMACRO
#define HLogger MultiDevicePlugin::Log::instance()
#endif


#define HLogPrint(isOn, isTraceCallStack, logLevel, level, tag, ...) \
    HLogger->doLog(isOn, isTraceCallStack, logLevel, level, __FILE__, __func__, __LINE__, tag, __VA_ARGS__)

// #define HFrequent(isOn, tag, ...) HLogPrint(isOn, MultiDevicePlugin::LogLevel::FREQUENT, "FREQ", tag, __VA_ARGS__)
// #define HFatal(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::FATAL, "FATAL", nullptr, __VA_ARGS__)
#define LOG_TRACE(isOn, tag, ...) HLogPrint(isOn, false, MultiDevicePlugin::LogLevel::PROCESS, "TRACE", tag, __VA_ARGS__)
#define LOG_DEBUG(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::DEBUG, "DEBUG", nullptr, __VA_ARGS__)
#define LOG_INFO(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::INFO, "INFO", nullptr, __VA_ARGS__)
#define LOG_WARNING(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::WARN, "WARN", nullptr, __VA_ARGS__)
#define LOG_ERROR(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::ERROR, "ERROR", nullptr, __VA_ARGS__)

#define LOG_DEBUG_TAG(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::DEBUG, "DEBUG", GetLogTag().c_str(), __VA_ARGS__)
#define LOG_INFO_TAG(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::INFO, "INFO", GetLogTag().c_str(), __VA_ARGS__)
#define LOG_WARNING_TAG(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::WARN, "WARN", GetLogTag().c_str(), __VA_ARGS__)
#define LOG_ERROR_TAG(...) HLogPrint(true, false, MultiDevicePlugin::LogLevel::ERROR, "ERROR", GetLogTag().c_str(), __VA_ARGS__)

#define TraceCallStacks(...) HLogPrint(true, true, MultiDevicePlugin::LogLevel::DEBUG, "DEBUG", nullptr, __VA_ARGS__)
#define TraceCallStack() TraceCallStacks(" ")

namespace MultiDevicePlugin {
inline bool setLogLevel(std::string logLevel) {
    static std::map<std::string, LogLevel> logValueMap = {{CONFIG_VALUE(LOG_NONE), LogLevel::LOG_NONE},
        {CONFIG_VALUE(LOG_ERROR), LogLevel::LOG_ERROR},
        {CONFIG_VALUE(LOG_WARNING), LogLevel::LOG_WARNING},
        {CONFIG_VALUE(LOG_INFO), LogLevel::LOG_INFO},
        {CONFIG_VALUE(LOG_DEBUG), LogLevel::LOG_DEBUG},
        {CONFIG_VALUE(LOG_TRACE), LogLevel::LOG_TRACE}};
    auto it = logValueMap.find(logLevel);
    if (it != logValueMap.end()) {
        HLogger->setLogLevel(it->second);
        return true;
    } else {
        return false;
    }
}

inline void INFO_RUN(const LogTask& task) {
   HLogger->doRun(MultiDevicePlugin::LogLevel::INFO, task);
}

inline void DEBUG_RUN(const LogTask& task) {
   HLogger->doRun(MultiDevicePlugin::LogLevel::DEBUG, task);
}

} // namespace MultiDevicePlugin

#endif //MULTIDEVICEPLUGIN_HLOG_H
