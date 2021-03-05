// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN32
#include <unistd.h>
#endif

#include <string>
#include <vector>
#include <map>
#include <xml_parse_utils.h>

#include "ie_itt.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "transformations/serialize.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "cpp/ie_cnn_network.h"
#include "details/ie_exception.hpp"

#include "ngraph/pass/pass.hpp"
#include "ngraph/variant.hpp"
#include "ngraph/function.hpp"
#include "ngraph/opsets/opset6.hpp"

#ifdef WIN32
#define stat _stat
#endif

namespace InferenceEngine {

struct NetworkCompilationContext final {
    static std::string calculateFileInfo(const std::string& filePath) {
        std::string res;
        struct stat result;
        size_t seed {};
        seed = hash_combine(seed, filePath);
        if (stat(filePath.c_str(), &result) == 0) {
            seed = hash_combine(seed, result.st_mtime);
            seed = hash_combine(seed, result.st_size);
        }
        return std::to_string(seed);
    }

    static std::string computeHash(const CNNNetwork& network,
                                   const std::map<std::string, std::string>& compileOptions) {
        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "NetworkCompilationContext::computeHash - CNN");
        std::stringstream xmlFile;
        std::stringstream binFile;

        // 1. Serialize
        CNNNetwork networkCopy = network; // TODO: it is not a real copy

        // TODO: add extensions (custom ops) to serializer
        ngraph::pass::Serialize serializer(xmlFile, binFile,
            ngraph::pass::Serialize::Version::IR_V10);
        serializer.run_on_function(networkCopy.getFunction());

        // 2. Compute hash on serialized data and options
        size_t seed {};
        seed = hash_combine(seed, xmlFile.str());
        seed = hash_combine(seed, binFile.str());
        for (const auto& kvp : compileOptions) {
            seed = hash_combine(seed, kvp.first + kvp.second);
        }
        return std::to_string(seed);
    }

    static std::string computeHash(const std::string& modelName,
                                   const std::map<std::string, std::string>& compileOptions) {
        OV_ITT_SCOPED_TASK(itt::domains::IE_LT, "NetworkCompilationContext::computeHash - ModelName");
        size_t seed {};
        seed = hash_combine(seed, modelName);
        for (const auto& kvp : compileOptions) {
            seed = hash_combine(seed, kvp.first + kvp.second);
        }
        return std::to_string(seed);
    }

private:
    template <typename T>
    static std::size_t hash_combine(std::size_t seed, const T& a) {
        std::size_t val = std::hash<T>()(a);

        // Hash combine formula from boost
        return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }
};

class CompiledBlobHeader final {
    std::string m_ieVersion;
    std::string m_fileInfo;

public:
    CompiledBlobHeader() = default;

    explicit CompiledBlobHeader(const std::string& ieVersion, const std::string& fileInfo) :
            m_ieVersion(ieVersion),
            m_fileInfo(fileInfo) {
    }

    const std::string& getIeVersion() const {
        return m_ieVersion;
    }

    const std::string& getFileInfo() const {
        return m_fileInfo;
    }

    friend std::istream & operator >> (std::istream& stream, CompiledBlobHeader& header) {
        std::string xmlStr;
        std::getline(stream, xmlStr);

        pugi::xml_document document;
        pugi::xml_parse_result res = document.load_string(xmlStr.c_str());

        if (res.status != pugi::status_ok) {
            THROW_IE_EXCEPTION_WITH_STATUS(NETWORK_NOT_READ) << "Error reading compiled blob header";
        }

        pugi::xml_node compiledBlobNode = document.document_element();
        header.m_ieVersion = XMLParseUtils::GetStrAttr(compiledBlobNode, "ie_version");
        header.m_fileInfo = XMLParseUtils::GetStrAttr(compiledBlobNode, "file_info");

        return stream;
    }

    friend std::ostream & operator << (std::ostream& stream, const CompiledBlobHeader& header) {
        pugi::xml_document document;
        auto compiledBlobNode = document.append_child("compiled_blob");
        compiledBlobNode.append_attribute("ie_version").set_value(header.m_ieVersion.c_str());
        compiledBlobNode.append_attribute("file_info").set_value(header.m_fileInfo.c_str());

        document.save(stream, nullptr, pugi::format_raw);
        document.reset();
        stream << std::endl;

        return stream;
    }
};

}  // namespace InferenceEngine
