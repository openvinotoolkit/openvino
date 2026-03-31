// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of a parallel memcpy streambuf for memory-backed reads.
 * @file parallel_mem_streambuf.hpp
 */

#pragma once

#include <memory>
#include <streambuf>

#include "openvino/util/parallel_read_streambuf.hpp"

namespace ov::util {

/**
 * @brief A std::streambuf that reads from an in-memory buffer using parallel
 *        memcpy for large reads.
 *
 * Intended for mmap-backed tensors: the tensor's raw memory is already mapped
 * into the process but pages may not yet be resident.  For large reads,
 * splitting the copy across N threads triggers concurrent page faults, raising
 * the OS I/O queue depth and saturating NVMe bandwidth.
 *
 * On Windows, after each large copy the consumed source pages are released
 * from the process working-set via VirtualFree(MEM_RESET) to relieve RAM
 * pressure when loading multi-GB models.
 *
 * Usage:
 * @code
 *   // In plugin::import_model(const ov::Tensor& model, ...):
 *   ov::util::ParallelMemStreamBuf par_buf(model.data(), model.get_byte_size());
 *   std::istream stream(&par_buf);
 *   // pass stream to BinaryInputBuffer or any std::istream& consumer
 * @endcode
 */
class ParallelMemStreamBuf : public std::streambuf {
public:
    /**
     * @param data       Pointer to the start of the memory region.
     * @param size       Total size of the memory region in bytes.
     * @param threshold  Minimum read size to engage parallel memcpy.
     */
    explicit ParallelMemStreamBuf(const void* data, size_t size, size_t threshold = DEFAULT_PARALLEL_IO_THRESHOLD);

    ~ParallelMemStreamBuf() override = default;

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

    const char* m_begin;     ///< start of the memory region
    const char* m_end;       ///< one-past-the-end of the memory region
    const char* m_current;   ///< current read position
    size_t m_threshold;
    /// Non-null when source is a file-backed mmap: delegates all I/O to
    /// ReadFile (Windows) / pread (Linux) parallel reads, bypassing the
    /// 2x RAM pressure and page-fault overhead of mmap+memcpy.
    std::unique_ptr<ParallelReadStreamBuf> m_file_buf;
};

}  // namespace ov::util