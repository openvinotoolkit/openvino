// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <streambuf>

#include "openvino/util/file_util.hpp"
#include "openvino/util/native_streambuf.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::intel_gpu {

// A temporary solution to improve SSD reading bandwidth and avoid the 2x-RAM cost of mmap+memcpy for mmap files.
// Once Level Zero handles SSD reading bandwidth issue, this can be removed and the original mmap+memcpy path can be restored.
//
// A std::streambuf that reads from an in-memory buffer using parallel
// memcpy for large reads.  When the source is a file-backed mmap, the
// constructor detects this and delegates to NativeStreamBuf limited to the
// same file region, so direct pread/ReadFile calls replace the 2x-RAM
// mmap+memcpy path without exposing data located after the region.
class ParallelMemStreamBuf : public std::streambuf {
public:
    explicit ParallelMemStreamBuf(const void* data, size_t size, size_t threshold = ov::util::default_parallel_io_threshold);

    ~ParallelMemStreamBuf() override;

    ParallelMemStreamBuf(const ParallelMemStreamBuf&) = delete;
    ParallelMemStreamBuf& operator=(const ParallelMemStreamBuf&) = delete;

protected:
    std::streamsize xsgetn(char_type* dst, std::streamsize n) override;
    int_type underflow() override;
    int_type uflow() override;
    pos_type seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which) override;
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;
    std::streamsize showmanyc() override;

private:
    void parallel_copy(char* dst, const char* src, size_t size);

    const char* m_begin;
    const char* m_end;
    const char* m_current;
    size_t m_threshold;
    FileHandle m_file_handle = ov::invalid_handle;  // owned; must outlive m_file_buf
    std::unique_ptr<ov::util::NativeStreamBuf> m_file_buf;
};

}  // namespace ov::intel_gpu
