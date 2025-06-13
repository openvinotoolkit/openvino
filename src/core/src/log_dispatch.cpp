// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/log_dispatch.hpp"

namespace ov::util {

LogStream::LogStream(std::ostream* default_ostream)
    : std::ostream(nullptr),
      default_outstream{default_ostream},
      default_streambuf{default_ostream->rdbuf()} {
    // default_outstream->rdbuf(nullptr);
}

LogStream::~LogStream() {
    default_outstream->rdbuf(default_streambuf);
}

LogStream& LogDispatch::Err() {
    static LogStream err_log{&std::cerr};
    return err_log;
}
LogStream& LogDispatch::Out() {
    static LogStream out_log{&std::cout};
    return out_log;
}
}  // namespace ov::util
