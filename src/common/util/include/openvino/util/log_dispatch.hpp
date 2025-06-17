// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <memory>

namespace ov::util {

class LogStream : public std::ostream {
public:
    LogStream(std::ostream* default_stream);

    LogStream(const LogStream&) = delete;
    LogStream(LogStream&&) = delete;
    LogStream& operator=(const LogStream&) = delete;
    LogStream& operator=(LogStream&&) = delete;

private:
    struct LogBuffer : std::streambuf {
        LogBuffer();
        int overflow(int c) override;
        std::ostream* current_ostream{};
    };
    LogBuffer log_buffer;

    std::ostream* const default_stream{};
};

class LogDispatch {
public:
    static LogStream& cerr();
    static LogStream& cout();

private:
    static LogStream err_log, out_log;
};

static auto& ov_cerr = LogDispatch::cerr();
static auto& ov_cout = LogDispatch::cout();
}  // namespace ov::util

using ov::util::ov_cerr;
using ov::util::ov_cout;
