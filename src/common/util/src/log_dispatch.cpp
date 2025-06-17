// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/log_dispatch.hpp"

namespace ov::util {

LogStream::LogStream(std::ostream* const def_os)
    : std::ostream{nullptr},
      default_ostream{def_os},
      current_ostream{def_os} {
    log_buffer.current_ostream = current_ostream;
    std::ostream::rdbuf(&log_buffer);
}

LogStream::LogBuffer::LogBuffer() : std::streambuf{} {}

int LogStream::LogBuffer::overflow(int c) {
    // Such performance dropping buffer address reading is needed for current testing approach, which replaces
    // cout/cerr `streambuf` with `stringbuf` for runtime comparison.
    current_ostream->rdbuf()->sputc(c);
    return c;
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
