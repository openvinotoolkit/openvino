//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/logger.hpp"

#include <iostream>

LogLevel Logger::global_lvl = LogLevel::None;

Logger::Logger(LogLevel lvl): m_lvl(lvl) {
}

std::stringstream& Logger::stream() {
    return m_ss;
}

Logger::~Logger() {
    if (m_lvl <= Logger::global_lvl) {
        switch (m_lvl) {
        case LogLevel::Info:
            std::cout << "[ INFO ] " << m_ss.str();
            break;
        case LogLevel::Debug:
            std::cout << "[ DEBUG ] " << m_ss.str();
            break;
        default:
                /* do nothing */;
        }
    }
}
