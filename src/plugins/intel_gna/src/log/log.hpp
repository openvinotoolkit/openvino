// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <iostream>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace intel_gna {

/**
 * @brief Log manager. It supports all ov::log::Level values.
 * To use it need to set log level then just call one of the logging methods:
 *      GnaLog::Gnalog(ov::log::Level::DEBUG);
 *      GnaLog::info()  << "log message"; // this message will be printed
 *      GnaLog::trace() << "log message"; // this message shoudl not be printed
 */
class GnaLog {
    GnaLog() = default;

    static GnaLog& log(ov::log::Level log_level) {
        GnaLog& obj = get_instance();
        obj.message_level_ = log_level;
        obj << "[" << log_level << "]" << " ";
        return obj;
    }

    /** Configuration log level. It should be set in the plugin configuration */
    ov::log::Level log_level_ = ov::log::Level::NO;

    /** Log level of particular log message */
    ov::log::Level message_level_ = ov::log::Level::NO;

    static GnaLog& get_instance() {
        static GnaLog log_obj;
        return log_obj;
    }

 public :
    GnaLog(const GnaLog&) = delete;
    void operator = (const GnaLog&) = delete;

    static void set_log_level(ov::log::Level log_level) {
        get_instance().log_level_ = log_level;
    }

    static ov::log::Level get_log_level() {
        return get_instance().log_level_;
    }

    /**
     * @brief Set ERROR log level
     * @return GnaLog object
     */
    static GnaLog& error() {
        return log(ov::log::Level::ERR);
    }

    /**
     * @brief Set WARNING log level
     * @return GnaLog object
     */
    static GnaLog& warning() {
        return log(ov::log::Level::WARNING);
    }

    /**
     * @brief Set DEBUG log level
     * @return GnaLog object
     */
    static GnaLog& debug() {
        return log(ov::log::Level::DEBUG);
    }

    /**
     * @brief Set INFO log level
     * @return GnaLog object
     */
    static GnaLog& info() {
        return log(ov::log::Level::INFO);
    }

    /**
     * @brief Set TRACE log level
     * @return GnaLog object
     */
    static GnaLog& trace() {
        return log(ov::log::Level::TRACE);
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

template <class T>
std::string toHexString(T t) {
    std::ostringstream o;
    o << std::hex << t;
    return o.str();
}

// alias for GnaLog class to make it aligned with snake style code
using log = GnaLog;

}  // namespace intel_gna
}  // namespace ov