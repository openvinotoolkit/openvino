// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <map>
#include <memory>
#include <ostream>
#include <string>

#include "openvino/core/any.hpp"

namespace ov {

class Tensor;
class Model;

struct ModelCache final {
    static std::string calculate_file_info(const std::string& filePath);

    static std::string compute_hash(const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& compileOptions);

    static std::string compute_hash(const std::string& modelName, const ov::AnyMap& compileOptions);
    static std::string compute_hash(const std::string& modeStr,
                                    const ov::Tensor& data,
                                    const ov::AnyMap& compileOptions);
};

class CompiledBlobHeader final {
    std::string m_ieVersion;
    std::string m_fileInfo;

public:
    CompiledBlobHeader();
    CompiledBlobHeader(const std::string& ieVersion, const std::string& fileInfo);

    const std::string& getIeVersion() const {
        return m_ieVersion;
    }

    const std::string& getFileInfo() const {
        return m_fileInfo;
    }

    friend std::istream& operator>>(std::istream& stream, CompiledBlobHeader& header);

    friend std::ostream& operator<<(std::ostream& stream, const CompiledBlobHeader& header);
};

}  // namespace ov
