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
        std::ostream* m_current_ostream{};
    };
    LogBuffer m_log_buffer;

    std::ostream* const m_default_stream{};
};

class LogDispatch {
public:
    static LogStream& cerr();
    static LogStream& cout();
};

static auto& ov_cerr = LogDispatch::cerr();
static auto& ov_cout = LogDispatch::cout();
}  // namespace ov::util

using ov::util::ov_cerr;
using ov::util::ov_cout;
