// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with logging facility for common samples
 * @file log.hpp
 */

#pragma once

#include <string>

namespace slog {

/**
 * @class LogStreamEndLine
 * @brief The LogStreamEndLine class implements an end line marker for a log stream
 */
class LogStreamEndLine { };

static constexpr LogStreamEndLine endl;


/**
 * @class LogStreamBoolAlpha
 * @brief The LogStreamBoolAlpha class implements bool printing for a log stream
 */
class LogStreamBoolAlpha { };

static constexpr LogStreamBoolAlpha boolalpha;


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
    LogStream(const std::string &prefix, std::ostream& log_stream)
            : _prefix(prefix), _new_line(true) {
        _log_stream = &log_stream;
    }

    /**
     * @brief A stream output operator to be used within the logger
     * @param arg Object for serialization in the logger message
     */
    template<class T>
    LogStream &operator<<(const T &arg) {
        if (_new_line) {
            (*_log_stream) << "[ " << _prefix << " ] ";
            _new_line = false;
        }

        (*_log_stream) << arg;
        return *this;
    }

    // Specializing for LogStreamEndLine to support slog::endl
    LogStream& operator<< (const LogStreamEndLine &/*arg*/) {
        _new_line = true;

        (*_log_stream) << std::endl;
        return *this;
    }

    // Specializing for LogStreamBoolAlpha to support slog::boolalpha
    LogStream& operator<< (const LogStreamBoolAlpha &/*arg*/) {
        (*_log_stream) << std::boolalpha;
        return *this;
    }
};


static LogStream info("INFO", std::cout);
static LogStream warn("WARNING", std::cout);
static LogStream err("ERROR", std::cerr);

}  // namespace slog
