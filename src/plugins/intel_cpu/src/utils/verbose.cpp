// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "utils/general_utils.h"
#ifdef CPU_DEBUG_CAPS

#    include <node.h>

#    include <cstdlib>
#    include <iostream>
#    include <sstream>
#    include <string>

#    include "../src/common/c_types_map.hpp"
#    include "../src/common/verbose.hpp"
#    include "cpu_types.h"
#    include "dnnl_debug.h"
#    include "dnnl_types.h"
#    include "memory_desc/cpu_memory_desc_utils.h"
#    include "verbose.h"

namespace ov::intel_cpu {

bool Verbose::shouldBePrinted() const {
    if (lvl < 1) {
        return false;
    }

    if (lvl < 2 && one_of(node->getType(), Type::Input, Type::Output)) {
        return false;
    }

    if (lvl < 3 && node->isConstant()) {
        return false;
    }

    return true;
}

/**
 * Print node verbose execution information to cout.
 * Similiar to DNNL_VERBOSE output
 * Formating written in C using oneDNN format functions.
 * Can be rewritten in pure C++ if necessary
 */
void Verbose::printInfo() {
    enum Color : uint8_t { RED, GREEN, YELLOW, BLUE, PURPLE, CYAN };

    auto colorize = [&](const Color color, const std::string& str) {
        if (!colorUp) {
            return str;
        }

        const std::string red("\033[1;31m");
        const std::string green("\033[1;32m");
        const std::string yellow("\033[1;33m");
        const std::string blue("\033[1;34m");
        const std::string purple("\033[1;35m");
        const std::string cyan("\033[1;36m");
        const std::string reset("\033[0m");
        std::string colorCode;

        switch (color) {
        case RED:
            colorCode = red;
            break;
        case GREEN:
            colorCode = green;
            break;
        case YELLOW:
            colorCode = yellow;
            break;
        case BLUE:
            colorCode = blue;
            break;
        case PURPLE:
            colorCode = purple;
            break;
        case CYAN:
            colorCode = cyan;
            break;
        default:
            colorCode = reset;
            break;
        }

        return colorCode + str + reset;
    };

    // can be increased if necessary
    const int CPU_VERBOSE_DAT_LEN = 512;
    char portsInfo[CPU_VERBOSE_DAT_LEN] = {'\0'};
    int written = 0;
    int written_total = 0;

    auto shift = [&](int size) {
        if (written < 0 || written_total + size > CPU_VERBOSE_DAT_LEN) {
            const char* errorMsg = "# NOT ENOUGHT BUFFER SIZE #";
            snprintf(portsInfo, strlen(errorMsg) + 1, "%s", errorMsg);
            written_total = strlen(errorMsg);
            return;
        }

        written_total += size;
    };

    auto getFormatAndDims = [](const MemoryDescPtr& desc) -> std::pair<std::string, std::string> {
        if (DnnlExtensionUtils::ElementTypeToDataType(desc->getPrecision(), DnnlExtensionUtils::nothrow_tag{})) {
            if (auto dnnl_desc = MemoryDescUtils::convertToDnnlMemoryDesc(desc)->getDnnlDesc()) {
                using namespace dnnl::impl;
                auto fmt_str = md2fmt_str("", dnnl_desc.get(), format_kind_t::dnnl_format_kind_undef);
                auto dim_str = md2dim_str(dnnl_desc.get());
                return {fmt_str, dim_str};
            }
            return {"empty", {}};
        }
        auto fmt_str = desc->getPrecision().to_string();
        if (const auto& dims = desc->getShape().getDims(); !dims.empty()) {
            auto dim_str = dim2str(dims.front());
            std::for_each(++(dims.begin()), dims.end(), [&dim_str](size_t dim) {
                dim_str.append("x" + dim2str(dim));
            });
            return {fmt_str, dim_str};
        }
        return {fmt_str, {}};
    };

    auto formatMemDesc = [&](const MemoryDescPtr& desc, std::string& prefix) {
        prefix = colorize(BLUE, prefix);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, " ");
        shift(written);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, "%s", prefix.c_str());
        shift(written);
        const auto [fmt_str, dim_str] = getFormatAndDims(desc);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, "%s", fmt_str.c_str());
        shift(written);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, ":");
        shift(written);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, "%s", dim_str.c_str());
        shift(written);
    };

    for (size_t i = 0; i < node->getParentEdges().size(); i++) {
        std::string prefix("src:" + std::to_string(i) + ':');
        const auto& desc = node->getParentEdgeAt(i)->getMemory().getDescPtr();
        formatMemDesc(desc, prefix);
    }

    for (size_t i = 0; i < node->getChildEdges().size(); i++) {
        std::string prefix("dst:" + std::to_string(i) + ':');
        const auto& desc = node->getChildEdgeAt(i)->getMemory().getDescPtr();
        formatMemDesc(desc, prefix);
    }

    std::string post_ops;
    if (!node->getFusedWith().empty()) {
        post_ops += "post_ops:'";
        for (const auto& fusedNode : node->getFusedWith()) {
            post_ops.append(colorize(GREEN, fusedNode->getName()))
                .append(":")
                .append(colorize(CYAN, NameFromType(fusedNode->getType())))
                .append(":")
                .append(algToString(fusedNode->getAlgorithm()))
                .append(";");
        }
        post_ops += "'";
    }

    std::string nodeImplementer = "cpu";
    if (node->getType() == Type::Reference) {
        nodeImplementer = "ngraph_ref";  // ngraph reference
    }

    const std::string& nodeName = colorize(GREEN, node->getName());
    const std::string& nodeType = colorize(CYAN, NameFromType(node->getType()));
    const std::string& nodeAlg = algToString(node->getAlgorithm());
    const std::string& nodePrimImplType =
        impl_type_to_string(node->getSelectedPrimitiveDescriptor()->getImplementationType());

    stream << "ov_cpu_verbose" << ',' << "exec" << ',' << nodeImplementer << ',' << nodeName << ":" << nodeType << ":"
           << nodeAlg << ',' << nodePrimImplType << ',' << portsInfo << ',' << post_ops << ',';
}

void Verbose::printDuration() {
    const auto& duration = node->PerfCounter().duration().count();
    stream << duration << "ms";
}

void Verbose::flush() const {
    std::cout << stream.rdbuf() << "\n";
}

}  // namespace ov::intel_cpu

#endif  // CPU_DEBUG_CAPS
