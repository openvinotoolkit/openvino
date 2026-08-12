// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <variant>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

/**
 * @brief Class that allows reading from either an "std::istream" or "ov::Tensor" object using a single set of common
 * methods.
 */
class BlobSource {
public:
    BlobSource(std::istream& source, const ov::log::Level log_level = Logger::global().level());

    BlobSource(const ov::Tensor& source, const ov::log::Level log_level = Logger::global().level());

    void copy_from_source(void* destination, const size_t size);

    void* interpret_from_source(const size_t size);

    ov::Tensor get_roi_tensor_from_source(const size_t size);

    void move_cursor(const std::streamoff offset, const std::ios_base::seekdir reference = std::ios::beg);

    std::streampos get_cursor() const;

    size_t get_size() const;

private:
    std::variant<std::reference_wrapper<std::istream>, std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>
        m_source;
    size_t m_size;

    Logger m_logger;
};

}  // namespace intel_npu
