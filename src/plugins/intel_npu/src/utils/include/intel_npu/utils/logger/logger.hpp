// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Class for pretty-logging.
//

#pragma once

#include <iostream>
#include <sstream>

#include "openvino/runtime/properties.hpp"

namespace intel_npu {

//
// Logger
//

std::string printFormattedCStr(const char* fmt, ...)
#if defined(__clang__)
    ;
#elif defined(__GNUC__) || defined(__GNUG__)
    __attribute__((format(printf, 1, 2)));
#else
    ;
#endif

class Logger {
public:
    static Logger& global();

    Logger(std::string_view name, ov::log::Level lvl = ov::log::Level::NO);
    Logger(const Logger& log) = default;

    Logger clone(std::string_view name) const;

    auto name() const {
        return _name;
    }

    void setName(std::string_view name) {
        _name = name;
    }

    auto level() const {
        return _logLevel;
    }

    Logger& setLevel(ov::log::Level lvl) {
        _logLevel = lvl;
        return *this;
    }

    bool isActive(ov::log::Level msgLevel) const;

    static std::ostream& getBaseStream();
    static std::ostream& getLevelStream(ov::log::Level msgLevel);

    template <typename... Args>
    void error(const char* format, Args&&... args) const {
        addEntryPacked(ov::log::Level::ERR, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warning(const char* format, Args&&... args) const {
        addEntryPacked(ov::log::Level::WARNING, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void info(const char* format, Args&&... args) const {
        addEntryPacked(ov::log::Level::INFO, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void debug(const char* format, Args&&... args) const {
        addEntryPacked(ov::log::Level::DEBUG, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void trace(const char* format, Args&&... args) const {
        addEntryPacked(ov::log::Level::TRACE, format, std::forward<Args>(args)...);
    }

private:
    void addEntryPackedActive(ov::log::Level msgLevel, const std::string_view msg) const;

    template <typename... Args>
    void addEntryPacked(ov::log::Level msgLevel, const char* format, Args&&... args) const {
        if (!isActive(msgLevel)) {
            return;
        }
        addEntryPackedActive(msgLevel, printFormattedCStr(format, std::forward<Args>(args)...));
    }

    void addEntryPacked(ov::log::Level msgLevel, const char* msg) const {
        if (!isActive(msgLevel)) {
            return;
        }
        addEntryPackedActive(msgLevel, msg);
    }

private:
    std::string _name;
    ov::log::Level _logLevel = ov::log::Level::NO;
};

}  // namespace intel_npu
