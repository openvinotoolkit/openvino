// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with logging facility for common samples
 * @file log.hpp
 */

#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace slog {
/**
 * @class LogStreamEndLine
 * @brief The LogStreamEndLine class implements an end line marker for a log stream
 */
class LogStreamEndLine {};

static constexpr LogStreamEndLine endl;

/**
 * @class LogStreamBoolAlpha
 * @brief The LogStreamBoolAlpha class implements bool printing for a log stream
 */
class LogStreamBoolAlpha {};

static constexpr LogStreamBoolAlpha boolalpha;

/**
 * @class LogStreamFlush
 * @brief The LogStreamFlush class implements flushing for a log stream
 */
class LogStreamFlush {};

static constexpr LogStreamFlush flush;

/**
 * @class LogStream
 * @brief The LogStream class implements a stream for sample logging
 */
class LogStream {
    std::string _prefix;
    std::ostream* _log_stream;
    bool _new_line;

public:
    /**
     * @brief A constructor. Creates an LogStream object
     * @param prefix The prefix to print
     */
    LogStream(const std::string& prefix, std::ostream& log_stream);

    /**
     * @brief A stream output operator to be used within the logger
     * @param arg Object for serialization in the logger message
     */
    template <class T>
    LogStream& operator<<(const T& arg) {
        if (_new_line) {
            (*_log_stream) << "[ " << _prefix << " ] ";
            _new_line = false;
        }

        (*_log_stream) << arg;
        return *this;
    }

    /**
     * @brief Overload output stream operator to print vectors in pretty form
     * [value1, value2, ...]
     */
    template <typename T>
    LogStream& operator<<(const std::vector<T>& v) {
        (*_log_stream) << "[ ";

        for (auto&& value : v)
            (*_log_stream) << value << " ";

        (*_log_stream) << "]";

        return *this;
    }

    // Specializing for LogStreamEndLine to support slog::endl
    LogStream& operator<<(const LogStreamEndLine&);

    // Specializing for LogStreamBoolAlpha to support slog::boolalpha
    LogStream& operator<<(const LogStreamBoolAlpha&);

    // Specializing for LogStreamFlush to support slog::flush
    LogStream& operator<<(const LogStreamFlush&);
};

extern LogStream info;
extern LogStream warn;
extern LogStream err;

}  // namespace slog
