// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <streambuf>

#include "openvino/core/weights_prefetch.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::intel_gpu {

/**
 * @brief Read-only stream buffer over a tensor that drives a SequentialPrefetchScheduler when @p prefetch is set and
 * the tensor has memory hints, and is a plain memory reader otherwise.
 */
class PrefetchStreamBuf : public std::streambuf {
public:
    explicit PrefetchStreamBuf(const ov::Tensor& tensor, bool prefetch = true);

    void keep_read_data() noexcept {
        if (m_scheduler) {
            m_scheduler->keep_read_data();
        }
    }

protected:
    int_type underflow() override;
    int_type uflow() override;
    std::streamsize showmanyc() override;
    std::streamsize xsgetn(char_type* dst, std::streamsize count) override;
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override;
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;

private:
    // The base get area is unused: setg() takes non-const pointers and blobs may be read-only.
    const char* m_begin;
    const char* m_current;
    const char* m_end;
    std::unique_ptr<ov::util::SequentialPrefetchScheduler> m_scheduler;
};

}  // namespace ov::intel_gpu
