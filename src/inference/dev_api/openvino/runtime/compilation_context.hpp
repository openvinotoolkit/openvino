// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <map>
#include <memory>
#include <ostream>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

class OPENVINO_RUNTIME_API ModelCache {
public:
    static std::string calculate_file_info(const std::string& filePath);

    static std::string compute_hash(const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& compileOptions);

    static std::string compute_hash(const std::string& modelName, const ov::AnyMap& compileOptions);
    static std::string compute_hash(const std::string& modeStr,
                                    const ov::Tensor& data,
                                    const ov::AnyMap& compileOptions);
    static std::string compute_hash(const std::shared_ptr<const ov::Model>& model,
                                    const std::filesystem::path& model_path,
                                    const ov::AnyMap& compileOptions);
};

class CompiledBlobHeader final {
    std::string m_ieVersion;
    std::string m_fileInfo;
    std::string m_runtimeInfo;
    std::size_t m_headerSizeAlignment{0};

public:
    CompiledBlobHeader();
    CompiledBlobHeader(const std::string& ieVersion,
                       const std::string& fileInfo,
                       const std::string& runtimeInfo,
                       const std::size_t headerSizeAlignment = 0);

    const std::string& get_openvino_version() const {
        return m_ieVersion;
    }

    const std::string& get_file_info() const {
        return m_fileInfo;
    }

    const std::string& get_runtime_info() const {
        return m_runtimeInfo;
    }

    const size_t get_header_size_alignment() const {
        return m_headerSizeAlignment;
    }

    friend std::istream& operator>>(std::istream& stream, CompiledBlobHeader& header);

    friend std::ostream& operator<<(std::ostream& stream, const CompiledBlobHeader& header);

    void read_from_buffer(const char* buffer, size_t buffer_size, size_t& pos);
};

}  // namespace ov
