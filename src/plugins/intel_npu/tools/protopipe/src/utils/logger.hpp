//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>

enum class LogLevel {
    None = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
};

class Logger {
public:
    static LogLevel global_lvl;
    explicit Logger(LogLevel lvl);
    std::stringstream& stream();
    ~Logger();

private:
    LogLevel m_lvl;
    std::stringstream m_ss;
};

#define LOG_WARN() Logger{LogLevel::Warn}.stream()
#define LOG_INFO() Logger{LogLevel::Info}.stream()
#define LOG_DEBUG() Logger{LogLevel::Debug}.stream()
