// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <map>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/visibility.hpp"

namespace ov::util {

class OPENVINO_API OstreamHashWrapperBin final : public std::streambuf {
    uint64_t m_res = 0lu;

public:
    uint64_t get_result() const {
        return m_res;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override;
};

class OPENVINO_API ConstantWriter {
public:
    using FilePosition = int64_t;
    using HashValue = size_t;
    using ConstWritePositions = std::multimap<HashValue, std::pair<FilePosition, const void*>>;

    ConstantWriter(std::ostream& bin_data, bool enable_compression = true);
    virtual ~ConstantWriter();

    virtual FilePosition write(const char* ptr,
                               size_t size,
                               size_t& new_size,
                               bool compress_to_fp16 = false,
                               ov::element::Type src_type = ov::element::dynamic,
                               bool ptr_is_temporary = false);

private:
    static std::unique_ptr<char[]> compress_data_to_fp16(const char* ptr,
                                                         size_t size,
                                                         ov::element::Type src_type,
                                                         size_t& compressed_size);

    ConstWritePositions m_hash_to_file_positions;
    std::reference_wrapper<std::ostream> m_binary_output;
    bool m_enable_compression;
    bool m_write_hash_value;
    FilePosition m_blob_offset;  // blob offset inside output stream
};
}  // namespace ov::util
