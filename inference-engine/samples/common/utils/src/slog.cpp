// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "samples/slog.hpp"

#include <iostream>

namespace slog {

LogStream info("INFO", std::cout);
LogStream warn("WARNING", std::cout);
LogStream err("ERROR", std::cerr);

LogStream::LogStream(const std::string& prefix, std::ostream& log_stream): _prefix(prefix), _new_line(true) {
    _log_stream = &log_stream;
}

// Specializing for LogStreamEndLine to support slog::endl
LogStream& LogStream::operator<<(const LogStreamEndLine& /*arg*/) {
    _new_line = true;

    (*_log_stream) << std::endl;
    return *this;
}

// Specializing for LogStreamBoolAlpha to support slog::boolalpha
LogStream& LogStream::operator<<(const LogStreamBoolAlpha& /*arg*/) {
    (*_log_stream) << std::boolalpha;
    return *this;
}

}  // namespace slog