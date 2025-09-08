// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <streambuf>

namespace intel_npu::driver_compiler_utils {

/**
 *  @brief Counter stream buffer, just counts the written bytes.
 *  Reads will result in EOF and no seek is supported.
 */
class counter_streambuf final : public std::streambuf {
public:
    /// Return the number of bytes written to the stream
    std::streamsize size() {
        return m_size;
    }

private:
    virtual int overflow(int c) override {
        ++m_size;
        return c;
    }

    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        m_size += n;
        return n;
    }

    virtual std::streampos seekoff(std::streamoff off,
                                   std::ios_base::seekdir way,
                                   std::ios_base::openmode which) override {
        // Return current stream position
        if (off == 0 && way == std::ios_base::cur && which == std::ios_base::out) {
            return m_size;
        } else {
            // No seek support
            throw std::runtime_error("Seek operation is not supported for counting_streambuf");
        }
    }

    std::streamsize m_size = 0;
};

/**
 *  @brief Writer stream buffer, writes data to target iterator.
 *  Reads will result in EOF and no seek is supported.
 */
template <typename OutputIt>
class writer_streambuf final : public std::streambuf {
public:
    writer_streambuf(const OutputIt& it) : startIt(it), writeIt(it) {}

private:
    int overflow(int c) override {
        return *writeIt++ = c;
    }

    std::streamsize xsputn(const char* s, std::streamsize n) override {
        writeIt = std::copy_n(s, n, writeIt);
        return n;
    }

    std::streampos seekoff(std::streamoff off, std::ios_base::seekdir way, std::ios_base::openmode which) override {
        // Return current stream position
        if (off == 0 && way == std::ios_base::cur && which == std::ios_base::out) {
            return std::distance(startIt, writeIt);
        } else {
            // No seek support
            throw std::runtime_error("Seek operation is not supported for writer_streambuf");
        }
    }

    OutputIt startIt;
    OutputIt writeIt;
};

}  // namespace intel_npu::driver_compiler_utils
