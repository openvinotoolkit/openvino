// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/log_dispatch.hpp"

namespace ov::util {

LogStream::LogStream(std::ostream* const default_stream) : std::ostream{nullptr}, m_default_stream{default_stream} {
    m_log_buffer.m_current_ostream = m_default_stream;
    std::ostream::rdbuf(&m_log_buffer);
}

LogStream::LogBuffer::LogBuffer() : std::streambuf{} {}

int LogStream::LogBuffer::overflow(int c) {
    // Such performance dropping buffer address reading is needed for current testing approach, which replaces
    // cout/cerr `streambuf` with `stringbuf` for runtime comparison.
    m_current_ostream->rdbuf()->sputc(c);
    return c;
}

LogStream& LogDispatch::cerr() {
    static LogStream err_log{&std::cerr};
    return err_log;
}
LogStream& LogDispatch::cout() {
    static LogStream out_log{&std::cout};
    return out_log;
}
}  // namespace ov::util
