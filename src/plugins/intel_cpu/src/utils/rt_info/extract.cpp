// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract.hpp"

#include <openvino/core/node.hpp>
#include <utils/ngraph_utils.hpp>
#include "onednn/dnnl.h"
#include "onednn/iml_type_mapper.h"
#include <oneapi/dnnl/dnnl.hpp>

#include <string>
namespace ov {
namespace intel_cpu {
namespace rt_info {

std::string getOriginalLayerNames(const ov::Node::RTMap& rtInfo, const std::string& opName) {
    if (rtInfo.count("originalLayersNames")) {
        return getRTInfoValue(rtInfo, "originalLayersNames");
    }

    return opName;
}

std::vector<impl_desc_type> getPrimitivesPriority(const ov::Node::RTMap& rtInfo) {
    const std::string& primitivesPriority = getPrimitivesPriorityValue(rtInfo);
    if (primitivesPriority.empty())
        return {};

    std::vector <impl_desc_type> implPriorities;

    std::istringstream stream(primitivesPriority);
    std::string str;
    while (getline(stream, str, ',')) {
        if (str.substr(0, 4) != "cpu:")
            continue;

        const auto implName = parse_impl_name(str);
        if (implName == impl_desc_type::unknown)
            IE_THROW() << "Unsupported CPU implementation " << str;

        implPriorities.push_back(parse_impl_name(str));
    }

    return implPriorities;
}

std::vector<dnnl::memory::format_tag> getInputMemoryFormatsFilter(const ov::Node::RTMap& rtInfo) {
    const std::string inputMemoryFormats = getInputMemoryFormats(rtInfo);
    if (inputMemoryFormats.empty())
        return {};

    std::vector<dnnl::memory::format_tag> filter;

    std::istringstream stream(inputMemoryFormats);
    std::string str;
    while (getline(stream, str, ',')) {
        if (str.substr(0, 4) != "cpu:")
            continue;
        filter.push_back(dnnl::utils::str2fmt(str.substr(4, str.size()).c_str()));
    }

    return filter;
}

std::vector<dnnl::memory::format_tag> getOutputMemoryFormatsFilter(const ov::Node::RTMap& rtInfo) {
    const std::string outputMemoryFormats = getOutputMemoryFormats(rtInfo);
    if (outputMemoryFormats.empty())
        return {};

    std::vector<dnnl::memory::format_tag> filter;

    std::istringstream stream(outputMemoryFormats);
    std::string str;
    while (getline(stream, str, ',')) {
        if (str.substr(0, 4) != "cpu:")
            continue;
        filter.push_back(dnnl::utils::str2fmt(str.substr(4, str.size()).c_str()));
    }

    return filter;
}

bool shouldEnforceBF16evenForGraphTail(const ov::Node::RTMap& rtInfo) {
    const auto it = rtInfo.find("enforceBF16evenForGraphTail");
    if (it == rtInfo.end()) {
        return false;
    }

    return it->second.as<bool>();
}

}   // namespace rt_info
}   // namespace intel_cpu
}   // namespace ov
