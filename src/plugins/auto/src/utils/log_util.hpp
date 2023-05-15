// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef AUTOPLUGIN_HLOG_H
#define AUTOPLUGIN_HLOG_H

#include <cstdarg>

#include "log.hpp"
#include <ie_plugin_config.hpp>

#ifdef  MULTIUNITTEST
#include "include/mock_log_utils.hpp"
#define MOCKTESTMACRO virtual
#define auto_plugin mock_auto_plugin
#define HLogger MockMultiDevice::MockLog::GetInstance()
#else
#define MOCKTESTMACRO
#define HLogger auto_plugin::Log::instance()
#endif


#define HLogPrint(isOn, isTraceCallStack, logLevel, level, tag, ...) \
    HLogger->doLog(isOn, isTraceCallStack, logLevel, level, __FILE__, __func__, __LINE__, tag, __VA_ARGS__)

// #define HFrequent(isOn, tag, ...) HLogPrint(isOn, auto_plugin::LogLevel::FREQUENT, "FREQ", tag, __VA_ARGS__)
// #define HFatal(...) HLogPrint(true, false, auto_plugin::LogLevel::FATAL, "FATAL", nullptr, __VA_ARGS__)
#define LOG_TRACE(isOn, tag, ...) HLogPrint(isOn, false, auto_plugin::LogLevel::PROCESS, "TRACE", tag, __VA_ARGS__)
#define LOG_DEBUG(...) HLogPrint(true, false, auto_plugin::LogLevel::DEBUG, "DEBUG", nullptr, __VA_ARGS__)
#define LOG_INFO(...) HLogPrint(true, false, auto_plugin::LogLevel::INFO, "INFO", nullptr, __VA_ARGS__)
#define LOG_WARNING(...) HLogPrint(true, false, auto_plugin::LogLevel::WARN, "WARN", nullptr, __VA_ARGS__)
#define LOG_ERROR(...) HLogPrint(true, false, auto_plugin::LogLevel::ERROR, "ERROR", nullptr, __VA_ARGS__)

// To use macro LOG_XXX_TAG, need to implement get_log_tag() which returns log tag, the type of log tag is string
#define LOG_DEBUG_TAG(...) HLogPrint(true, false, auto_plugin::LogLevel::DEBUG, "DEBUG", get_log_tag().c_str(), __VA_ARGS__)
#define LOG_INFO_TAG(...) HLogPrint(true, false, auto_plugin::LogLevel::INFO, "INFO", get_log_tag().c_str(), __VA_ARGS__)
#define LOG_WARNING_TAG(...) HLogPrint(true, false, auto_plugin::LogLevel::WARN, "WARN", get_log_tag().c_str(), __VA_ARGS__)
#define LOG_ERROR_TAG(...) HLogPrint(true, false, auto_plugin::LogLevel::ERROR, "ERROR", get_log_tag().c_str(), __VA_ARGS__)

#define TraceCallStacks(...) HLogPrint(true, true, auto_plugin::LogLevel::DEBUG, "DEBUG", nullptr, __VA_ARGS__)
#define TraceCallStack() TraceCallStacks(" ")
namespace ov {
namespace auto_plugin {
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
   HLogger->doRun(auto_plugin::LogLevel::INFO, task);
}

inline void DEBUG_RUN(const LogTask& task) {
   HLogger->doRun(auto_plugin::LogLevel::DEBUG, task);
}

} // namespace auto_plugin
} // namespace ov
#endif //AUTOPLUGIN_HLOG_H
