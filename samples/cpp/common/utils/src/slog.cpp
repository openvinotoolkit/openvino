// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include <iostream>

#include "samples/slog.hpp"
// clang-format on

namespace slog {

LogStream info("INFO", std::cout);
LogStream warn("WARNING", std::cout);
LogStream err("ERROR", std::cerr);

LogStream::LogStream(const std::string& prefix, std::ostream& log_stream) : _prefix(prefix), _new_line(true) {
    _log_stream = &log_stream;
}

// Specializing for LogStreamEndLine to support slog::endl
LogStream& LogStream::operator<<(const LogStreamEndLine& /*arg*/) {
    if (_new_line)
        (*_log_stream) << "[ " << _prefix << " ] ";
    _new_line = true;

    (*_log_stream) << std::endl;
    return *this;
}

// Specializing for LogStreamBoolAlpha to support slog::boolalpha
LogStream& LogStream::operator<<(const LogStreamBoolAlpha& /*arg*/) {
    (*_log_stream) << std::boolalpha;
    return *this;
}

// Specializing for LogStreamFlush to support slog::flush
LogStream& LogStream::operator<<(const LogStreamFlush& /*arg*/) {
    (*_log_stream) << std::flush;
    return *this;
}

}  // namespace slog
