// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <memory>

#include "openvino/core/core_visibility.hpp"

namespace ov::util {

class OPENVINO_API LogStream : public std::ostream {
public:
    LogStream(std::ostream* default_ostream);
    ~LogStream();

    LogStream(const LogStream&) = delete;
    LogStream(LogStream&&) = delete;
    LogStream& operator=(const LogStream&) = delete;
    LogStream& operator=(LogStream&&) = delete;

private:
    std::ostream* default_outstream;
    std::streambuf* default_streambuf;
};

class OPENVINO_API LogDispatch {
public:
    static LogStream& Err();
    static LogStream& Out();

private:
    static LogStream err_log, out_log;
};

static auto& ov_cerr = LogDispatch::Err();
static auto& ov_cout = LogDispatch::Out();
}  // namespace ov::util

using ov::util::ov_cerr;
using ov::util::ov_cout;
