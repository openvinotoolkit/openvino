//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>

enum class LogLevel {
    None = 0,
    Info = 1,
    Debug = 2,
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

#define LOG_INFO() Logger{LogLevel::Info}.stream()
#define LOG_DEBUG() Logger{LogLevel::Debug}.stream()
