// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "dnnl_types.h"
#ifdef CPU_DEBUG_CAPS

#include "mkldnn_node.h"
#include "dnnl_debug.h"

#include <string>
#include <cstdlib>

namespace MKLDNNPlugin {
/**
 * Print node verbose execution information to cout.
 * Similiar to DNNL_VERBOSE output
 * Formating written in C using oneDNN format functions.
 * Can be rewritten in pure C++ if necessary
 */
void print(const MKLDNNNodePtr& node, const std::string& verboseLvl) {
    // use C atoi instead of std::stoi to avoid dealing with exceptions
    const int lvl = atoi(verboseLvl.c_str());

    if (lvl < 1)
        return;

    if (node->isConstant() ||
        node->getType() == Input || node->getType() == Output)
        return;

    /* 1,  2,  3,  etc -> no color
     * 11, 22, 33, etc -> colorize */
    bool colorUp = lvl / 10 > 0 ? true : false;
    // can be increased if necessary
    const int CPU_VERBOSE_DAT_LEN = 512;
    char portsInfo[CPU_VERBOSE_DAT_LEN] = {'\0'};
    int written = 0;
    int written_total = 0;

    enum Color {
        RED,
        GREEN,
        YELLOW,
        BLUE,
        PURPLE,
        CYAN
    };

    auto colorize = [&](const Color color, const std::string& str) {
        if (!colorUp)
            return str;

        const std::string     red("\033[1;31m");
        const std::string   green("\033[1;32m");
        const std::string  yellow("\033[1;33m");
        const std::string    blue("\033[1;34m");
        const std::string  purple("\033[1;35m");
        const std::string    cyan("\033[1;36m");
        const std::string   reset("\033[0m");
        std::string colorCode;

        switch (color) {
        case RED:    colorCode = red;
            break;
        case GREEN:  colorCode = green;
            break;
        case YELLOW: colorCode = yellow;
            break;
        case BLUE:   colorCode = blue;
            break;
        case PURPLE: colorCode = purple;
            break;
        case CYAN:   colorCode = cyan;
            break;
        default:     colorCode = reset;
            break;
        }

        return colorCode + str + reset;
    };

    auto shift = [&](int size) {
        if (written < 0 || written_total + size > CPU_VERBOSE_DAT_LEN) {
            const char* errorMsg = "# NOT ENOUGHT BUFFER SIZE #";
            snprintf(portsInfo, strlen(errorMsg) + 1, "%s", errorMsg);
            written_total = strlen(errorMsg);
            return;
        }

        written_total += size;
    };

    auto formatMemDesc = [&](const dnnl_memory_desc_t& desc, std::string& prefix) {
        prefix = colorize(BLUE, prefix);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, " ");
        shift(written);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, "%s", prefix.c_str());
        shift(written);
        written = dnnl_md2fmt_str(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, &desc);
        shift(written);
        written = snprintf(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, ":");
        shift(written);
        written = dnnl_md2dim_str(portsInfo + written_total, CPU_VERBOSE_DAT_LEN - written_total, &desc);
        shift(written);
    };

    for (int i = 0; i < node->getParentEdges().size(); i++) {
        std::string prefix("src:" + std::to_string(i) + ':');
        formatMemDesc(node->getParentEdgeAt(i)->getMemory().GetDescriptor().data, prefix);
    }

    for (int i = 0; i < node->getChildEdges().size(); i++) {
        std::string prefix("dst:" + std::to_string(i) + ':');
        formatMemDesc(node->getChildEdgeAt(i)->getMemory().GetDescriptor().data, prefix);
    }

    std::string post_ops;
    if (!node->getFusedWith().empty()) {
        post_ops += "post_ops:'";
        for (const auto& fusedNode : node->getFusedWith()) {
            post_ops.append(colorize(GREEN, fusedNode->getName())).append(":")
                .append(colorize(CYAN, NameFromType(fusedNode->getType()))).append(":")
                .append(algToString(fusedNode->getAlgorithm()))
                .append(";");
        }
        post_ops += "'";
    }

    const std::string nodeName = colorize(GREEN, node->getName());
    const std::string nodeType = colorize(CYAN, NameFromType(node->getType()));
    const std::string nodeAlg  = algToString(node->getAlgorithm());
    const std::string nodePrimImplType =  impl_type_to_string(node->getSelectedPrimitiveDescriptor()->getImplementationType());
    const auto duration = node->PerfCounter().duration().count();

    // Example:
    std::cout<< "ov_cpu_verbose,exec,cpu,"
              << nodeName << ":" << nodeType << ":" << nodeAlg << ','
              << nodePrimImplType << ','
              << portsInfo << ','
              << post_ops << ','
              << duration
              << "\n";
}
} // namespace MKLDNNPlugin
#endif // CPU_DEBUG_CAPS
