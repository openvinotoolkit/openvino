// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/logger/logger.hpp"

#include <cassert>
#include <chrono>
#include <cstdarg>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>

#define DEFAULT_COLOR "\033[0m"
#define RED           "\033[31m"
#define GREEN         "\033[32m"
#define YELLOW        "\033[33m"
#define BLUE          "\033[34m"
#define CYAN          "\033[36m"

namespace intel_npu {

std::string printFormattedCStr(const char* fmt, ...) {
    // Process with original buffer
    const int bufferSize = 256;
    std::va_list args;
    va_start(args, fmt);
    std::va_list argsForFinalBuffer;
    va_copy(argsForFinalBuffer, args);
    char buffer[bufferSize];
    auto requiredBytes = vsnprintf(buffer, bufferSize, fmt, args);
    va_end(args);

    if (requiredBytes < 0) {
        va_end(argsForFinalBuffer);
        return std::string("vsnprintf got error from fmt: ") + fmt;
    } else if (requiredBytes > bufferSize) {
        std::string out(requiredBytes, 0);  // +1 implicitly
        vsnprintf(out.data(), requiredBytes + 1, fmt, argsForFinalBuffer);
        va_end(argsForFinalBuffer);
        return out;
    }

    va_end(argsForFinalBuffer);
    return buffer;
}

//
// Logger
//
static const char* logLevelPrintout[] = {"NONE", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"};

Logger& Logger::global() {
#if defined(NPU_PLUGIN_DEVELOPER_BUILD) || !defined(NDEBUG)
    ov::log::Level logLvl = ov::log::Level::WARNING;
    if (const auto env = std::getenv("OV_NPU_LOG_LEVEL")) {
        try {
            std::istringstream is(env);
            is >> logLvl;
        } catch (...) {
            // Use deault log level
        }
    }
    static Logger log("global", logLvl);
#else
    static Logger log("global", ov::log::Level::ERR);
#endif
    return log;
}

Logger::Logger(std::string_view name, ov::log::Level lvl) : _name(name), _logLevel(lvl) {}

Logger Logger::clone(std::string_view name) const {
    Logger logger(name, level());
    return logger;
}

bool Logger::isActive(ov::log::Level msgLevel) const {
    return static_cast<int32_t>(msgLevel) <= static_cast<int32_t>(_logLevel);
}

std::ostream& Logger::getBaseStream() {
    return std::cout;
}

namespace {

std::ostream& getColor(ov::log::Level msgLevel) {
    std::ostream& stream = std::cout;

    switch (msgLevel) {
    case ov::log::Level::ERR:
        stream << RED;
        break;
    case ov::log::Level::WARNING:
        stream << YELLOW;
        break;
    case ov::log::Level::INFO:
        stream << CYAN;
        break;
    case ov::log::Level::DEBUG:
        stream << GREEN;
        break;
    case ov::log::Level::TRACE:
        stream << BLUE;
        break;
    default:
        stream << DEFAULT_COLOR;
    }
    return stream;
}

}  // namespace

std::ostream& Logger::getLevelStream(ov::log::Level msgLevel) {
    return getColor(msgLevel);
}

void Logger::addEntryPackedActive(ov::log::Level msgLevel, std::string_view msg) const {
    std::stringstream tempStream;
    char timeStr[] = "undefined_time";
    time_t now = time(nullptr);
    struct tm* loctime = localtime(&now);
    if (loctime != nullptr) {
        strftime(timeStr, sizeof(timeStr), "%H:%M:%S", loctime);
    }

    using namespace std::chrono;
    uint32_t ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() % 1000;
    auto& stream = getLevelStream(msgLevel);
    try {
        tempStream << "[" << logLevelPrintout[static_cast<int32_t>(msgLevel) + 1] << "] " << timeStr << "." << ms
                   << " [" << _name << "] ";

        tempStream << msg;

        static std::mutex logMtx;
        std::lock_guard<std::mutex> logMtxLock(logMtx);
        stream << tempStream.str() << DEFAULT_COLOR;
        stream << std::endl;
        stream.flush();
    } catch (const std::exception& e) {
        std::cerr << "Exception caught in Logger::addEntryPackedActive - " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened in Logger::addEntryPackedActive" << std::endl;
    }
    stream.flush();
}

}  // namespace intel_npu
