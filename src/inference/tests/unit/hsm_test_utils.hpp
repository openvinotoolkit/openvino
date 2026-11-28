// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ios>
#include <sstream>
#include <streambuf>

// Stream buffers shared by the HSM writer tests.
namespace ov::test {

// Fails (writing only half the requested bytes) on a chosen xsputn() call, simulating a stream write
// that partially succeeds before std::ostream sets badbit - built on std::stringbuf so seeking still
// works, letting a retry or a rollback be observed.
class FlakyOnceStreamBuf : public std::stringbuf {
public:
    int fail_on_call = -1;  // -1 = never fail
    int call_count = 0;

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (++call_count == fail_on_call) {
            const auto partial = n / 2;
            std::stringbuf::xsputn(s, partial);
            return partial;
        }
        return std::stringbuf::xsputn(s, n);
    }
};

// Ignores the put position: every write lands at the end, so a seek-back-and-patch goes to the wrong place.
class AppendOnlyStreamBuf : public std::stringbuf {
protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        seekoff(0, std::ios::end, std::ios::out);
        return std::stringbuf::xsputn(s, n);
    }
};

// The base class's seekoff/seekpos always fail, so tellp() reports -1.
struct NonSeekableStreamBuf : std::streambuf {};

}  // namespace ov::test
