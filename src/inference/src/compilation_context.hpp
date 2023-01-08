// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <map>
#include <ostream>
#include <string>

namespace InferenceEngine {

class CNNNetwork;

struct NetworkCompilationContext final {
    static std::string calculateFileInfo(const std::string& filePath);

    static std::string computeHash(const CNNNetwork& network, const std::map<std::string, std::string>& compileOptions);

    static std::string computeHash(const std::string& modelName,
                                   const std::map<std::string, std::string>& compileOptions);
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

}  // namespace InferenceEngine
