// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <iostream>

#include "openvino/runtime/properties.hpp"

/**
 * @brief Log manager. It supports all ov::log::Level values.
 * To use it need to set log level then just call one of the logging methods:
 *      GnaLog::Gnalog(ov::log::Level::DEBUG);
 *      GnaLog::LogInfo()  << "log message"; // this message will be printed
 *      GnaLog::LogTrace() << "log message"; // this message shoudl not be printed 
 */
class GnaLog {

 private :
    GnaLog() {}

    static GnaLog& Log(ov::log::Level log_level) {
        GnaLog& obj = getInstance();
        obj.message_level_ = log_level;
        obj << "[" << log_level << "]" << " ";
        return obj;
    };
    
    /** Configuration log level. It should be set in the plugin configuration */
    ov::log::Level log_level_ = ov::log::Level::NO;
    
    /** Log level of particular log message */
    ov::log::Level message_level_ = ov::log::Level::NO;

 public :
    GnaLog(GnaLog const&) = delete;
    void operator = (const GnaLog&) = delete;
    
    GnaLog(ov::log::Level log_level) {
        getInstance().log_level_ = log_level;
    }

    static GnaLog& getInstance()
    {
        static GnaLog log_obj;
        return log_obj;
    }

    /**
     * @brief Set ERROR log level
     * @return GnaLog object
     */
    static GnaLog& LogErr() {
        return Log(ov::log::Level::ERR);
    }

    /**
     * @brief Set WARNING log level
     * @return GnaLog object
     */
    static GnaLog& LogWarn() {
        return Log(ov::log::Level::WARNING);
    }

    /**
     * @brief Set DEBUG log level
     * @return GnaLog object
     */
    static GnaLog& LogDebug() {
        return Log(ov::log::Level::DEBUG);
    }

    /**
     * @brief Set INFO log level
     * @return GnaLog object
     */
    static GnaLog& LogInfo() {
        return Log(ov::log::Level::INFO);
    }

    /**
     * @brief Set TRACE log level
     * @return GnaLog object
     */
    static GnaLog& LogTrace() {
        return Log(ov::log::Level::TRACE);
    }

    template <class T>
    GnaLog &operator << (const T &obj) {
        if (message_level_ <= log_level_) {
            if (message_level_ == ov::log::Level::ERR) {
                std::cerr << obj;
            } else {
                std::cout << obj;
            }
        }
        return *this;
    }

    GnaLog &operator << (std::ostream & (*manip)(std::ostream &)) {
        if (message_level_ <= log_level_) {
            if (message_level_ == ov::log::Level::ERR) {
                std::cerr << manip;
            } else {
                std::cout << manip;
            }
        }
        return *this;
    }
};