// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cstdarg>

#include "log.hpp"

#ifdef  MULTIUNITTEST
#include "include/mock_log_utils.hpp"
#define MOCKTESTMACRO virtual
#define auto_plugin mock_auto_plugin
#define HLogger ov::mock_auto_plugin::MockLog::get_instance()
#else
#define MOCKTESTMACRO
#define HLogger ov::auto_plugin::Log::instance()
#endif


#define HLogPrint(isOn, isTraceCallStack, logLevel, level, tag, ...) \
    HLogger->do_log(isOn, isTraceCallStack, logLevel, level, __FILE__, __func__, __LINE__, tag, __VA_ARGS__)

// #define HFrequent(isOn, tag, ...) HLogPrint(isOn, auto_plugin::LogLevel::FREQUENT, "FREQ", tag, __VA_ARGS__)
// #define HFatal(...) HLogPrint(true, false, auto_plugin::LogLevel::FATAL, "FATAL", nullptr, __VA_ARGS__)
#define LOG_TRACE(isOn, tag, ...) HLogPrint(isOn, false, ov::log::Level::TRACE, "TRACE", tag, __VA_ARGS__)
#define LOG_DEBUG(...) HLogPrint(true, false, ov::log::Level::DEBUG, "DEBUG", nullptr, __VA_ARGS__)
#define LOG_INFO(...) HLogPrint(true, false, ov::log::Level::INFO, "INFO", nullptr, __VA_ARGS__)
#define LOG_WARNING(...) HLogPrint(true, false, ov::log::Level::WARNING, "WARN", nullptr, __VA_ARGS__)
#define LOG_ERROR(...) HLogPrint(true, false, ov::log::Level::ERR, "ERROR", nullptr, __VA_ARGS__)

// To use macro LOG_XXX_TAG, need to implement get_log_tag() which returns log tag, the type of log tag is string
#define LOG_DEBUG_TAG(...) HLogPrint(true, false, ov::log::Level::DEBUG, "DEBUG", get_log_tag().c_str(), __VA_ARGS__)
#define LOG_INFO_TAG(...) HLogPrint(true, false, ov::log::Level::INFO, "INFO", get_log_tag().c_str(), __VA_ARGS__)
#define LOG_WARNING_TAG(...) HLogPrint(true, false, ov::log::Level::WARNING, "WARN", get_log_tag().c_str(), __VA_ARGS__)
#define LOG_ERROR_TAG(...) HLogPrint(true, false, ov::log::Level::ERR, "ERROR", get_log_tag().c_str(), __VA_ARGS__)

#define TraceCallStacks(...) HLogPrint(true, true, ov::log::Level::DEBUG, "DEBUG", nullptr, __VA_ARGS__)
#define TraceCallStack() TraceCallStacks(" ")
namespace ov {
namespace auto_plugin {
inline bool set_log_level(ov::Any loglevel) {
    try {
        HLogger->set_log_level(loglevel.as<ov::log::Level>());
        return true;
    } catch (ov::Exception&) {
        return false;
    }
}

inline void INFO_RUN(const LogTask& task) {
   HLogger->do_run(ov::log::Level::INFO, task);
}

inline void DEBUG_RUN(const LogTask& task) {
   HLogger->do_run(ov::log::Level::DEBUG, task);
}

} // namespace auto_plugin
} // namespace ov
