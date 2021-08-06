// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compilation_context.hpp"

#include <sys/types.h>
#include <sys/stat.h>

#ifndef WIN32
#include <unistd.h>
#endif
#include <xml_parse_utils.h>

#include "ie_itt.hpp"
#include "transformations/serialize.hpp"
#include "cpp/ie_cnn_network.h"
#include "details/ie_exception.hpp"

#include "ngraph/variant.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "transformations/rt_info/dequantization_attribute.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "file_utils.h"

#ifdef WIN32
#define stat _stat
#endif

namespace InferenceEngine {

template <typename T>
static std::size_t hash_combine(std::size_t seed, const T& a) {
    // Hash combine formula from boost
    return seed ^ (std::hash<T>()(a) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template <typename T>
static int32_t as_int32_t(T v) {
    return static_cast<int32_t>(v);
}

class OstreamHashWrapper final: public std::streambuf {
    std::size_t m_res = 0;
public:
    std::size_t getResult() const { return m_res; }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        const std::int64_t* intS = (const std::int64_t *)s;
        std::streamsize n64 = n / sizeof(std::int64_t);
        std::streamsize i = 0;
        // Using 64-bit values executes much faster than char
        while (i++ < n64) {
            m_res += *(intS++);
        }

        std::streamsize rest = n % sizeof(std::int64_t);
        for (i = 0; i < rest; i++) {
            m_res += s[n - rest + i];
        }
        return n;
    }
};

//////////////////////////////////////////////////

std::string NetworkCompilationContext::calculateFileInfo(const std::string& filePath) {
    size_t seed = 0;
    auto absPath = filePath;
    try {
        absPath = FileUtils::absoluteFilePath(filePath);
    } catch (...) {
        // can't get absolute path, will use filePath for hash
    }

    seed = hash_combine(seed, absPath);

    std::string res;
    struct stat result;
    if (stat(absPath.c_str(), &result) == 0) {
        seed = hash_combine(seed, result.st_mtime);
        seed = hash_combine(seed, result.st_size);
    }
    return std::to_string(seed);
}

std::string NetworkCompilationContext::computeHash(const CNNNetwork& network,
                               const std::map<std::string, std::string>& compileOptions) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::IE_LT, "NetworkCompilationContext::computeHash - CNN");
    OstreamHashWrapper xmlHash;
    OstreamHashWrapper binHash;
    std::ostream xml(&xmlHash);
    std::ostream bin(&binHash);

    IE_ASSERT(network.getFunction());

    // 1. Serialize
    CNNNetwork net(network);
    ngraph::pass::Serialize serializer(xml, bin,
        ngraph::pass::Serialize::Version::IR_V10);
    serializer.run_on_function(net.getFunction());

    // 2. Compute hash on serialized data and options
    size_t seed = 0;
    seed = hash_combine(seed, xmlHash.getResult());
    seed = hash_combine(seed, binHash.getResult());

    for (const auto& kvp : compileOptions) {
        seed = hash_combine(seed, kvp.first + kvp.second);
    }

    // 3. Add runtime information which may not be serialized
    for (const auto& op : network.getFunction()->get_ordered_ops()) {
        const auto& rt = op->get_rt_info();
        for (const auto& rtMapData : rt) {
            seed = hash_combine(seed, rtMapData.first);

            if (auto stringData = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::string>>(rtMapData.second)) {
                seed = hash_combine(seed, stringData->get());
            } else if (auto intData = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::int64_t>>(rtMapData.second)) {
                seed = hash_combine(seed, intData->get());
            } else if (auto deq = std::dynamic_pointer_cast<ngraph::VariantWrapper<ngraph::DequantizationAttr>>(rtMapData.second)) {
                seed = hash_combine(seed, deq->get().getDequantizationAttr());
            } else if (auto fNames = std::dynamic_pointer_cast<ngraph::VariantWrapper<ngraph::FusedNames>>(rtMapData.second)) {
                seed = hash_combine(seed, fNames->get().getNames());
            } else if (auto prim = std::dynamic_pointer_cast<ngraph::VariantWrapper<ngraph::PrimitivesPriority>>(rtMapData.second)) {
                seed = hash_combine(seed, prim->get().getPrimitivesPriority());
            }
        }
    }

    // 4. Add inputs info
    for (const auto& input : network.getInputsInfo()) {
        InputInfo::Ptr info = input.second;
        seed = hash_combine(seed, as_int32_t(info->getPrecision()));
        seed = hash_combine(seed, as_int32_t(info->getLayout()));

        const InferenceEngine::PreProcessInfo& preproc = info->getPreProcess();
        seed = hash_combine(seed, as_int32_t(preproc.getMeanVariant()));

        if (preproc.getMeanVariant() == MeanVariant::MEAN_VALUE) {
            seed = hash_combine(seed, preproc.getNumberOfChannels());
            for (size_t c = 0; c < preproc.getNumberOfChannels(); ++c) {
                const PreProcessChannel::Ptr & channelInfo = preproc[c];
                seed = hash_combine(seed, channelInfo->stdScale);
                seed = hash_combine(seed, channelInfo->meanValue);
            }
        } else if (preproc.getMeanVariant() == MeanVariant::MEAN_IMAGE) {
            // TODO: think if we need to compute hash for mean image if it exists
        }
    }

    // 5. Add outputs info
    for (const auto& output : network.getOutputsInfo()) {
        DataPtr info = output.second;
        seed = hash_combine(seed, as_int32_t(info->getPrecision()));
        seed = hash_combine(seed, as_int32_t(info->getLayout()));
    }

    return std::to_string(seed);
}

std::string NetworkCompilationContext::computeHash(const std::string& modelName,
                               const std::map<std::string, std::string>& compileOptions) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::IE_LT, "NetworkCompilationContext::computeHash - ModelName");
    size_t seed = 0;
    try {
        seed = hash_combine(seed, FileUtils::absoluteFilePath(modelName));
    } catch (...) {
        // can't get absolute path, use modelName for hash calculation
        seed = hash_combine(seed, modelName);
    }
    for (const auto& kvp : compileOptions) {
        seed = hash_combine(seed, kvp.first + kvp.second);
    }
    return std::to_string(seed);
}

//////////////////////////////////////////////////

CompiledBlobHeader::CompiledBlobHeader() {}

CompiledBlobHeader::CompiledBlobHeader(const std::string& ieVersion, const std::string& fileInfo) :
        m_ieVersion(ieVersion),
        m_fileInfo(fileInfo) {
}

std::istream& operator >> (std::istream& stream, CompiledBlobHeader& header) {
    std::string xmlStr;
    std::getline(stream, xmlStr);

    pugi::xml_document document;
    pugi::xml_parse_result res = document.load_string(xmlStr.c_str());

    if (res.status != pugi::status_ok) {
        IE_THROW(NetworkNotRead) << "Error reading compiled blob header";
    }

    pugi::xml_node compiledBlobNode = document.document_element();
    header.m_ieVersion = XMLParseUtils::GetStrAttr(compiledBlobNode, "ie_version");
    header.m_fileInfo = XMLParseUtils::GetStrAttr(compiledBlobNode, "file_info");

    return stream;
}

std::ostream& operator << (std::ostream& stream, const CompiledBlobHeader& header) {
    pugi::xml_document document;
    auto compiledBlobNode = document.append_child("compiled_blob");
    compiledBlobNode.append_attribute("ie_version").set_value(header.m_ieVersion.c_str());
    compiledBlobNode.append_attribute("file_info").set_value(header.m_fileInfo.c_str());

    document.save(stream, nullptr, pugi::format_raw);
    document.reset();
    stream << std::endl;

    return stream;
}

}  // namespace InferenceEngine
