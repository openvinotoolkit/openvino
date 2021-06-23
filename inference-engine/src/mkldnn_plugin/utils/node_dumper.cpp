// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef CPU_DEBUG_CAPS

#include "node_dumper.h"

#include "mkldnn_node.h"
#include "ie_common.h"
#include "utils/blob_dump.h"
#include "utils/debug_capabilities.h"

#include <array>
#include <regex>
#include <sstream>
#include <string>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

NodeDumper::NodeDumper(const DebugCaps::Config& config, const int _count)
    : dumpFormat(FORMAT::BIN)
    , dumpDirName("mkldnn_dump")
    , count(_count) {
    if (!config.blobDumpDir.empty())
        dumpDirName = config.blobDumpDir;

    if (!config.blobDumpFormat.empty())
        dumpFormat = parseDumpFormat(config.blobDumpFormat);

    if (!config.blobDumpNodeExecId.empty())
        dumpFilters[FILTER::BY_EXEC_ID] = config.blobDumpNodeExecId;

    if (!config.blobDumpNodePorts.empty())
        dumpFilters[FILTER::BY_PORTS] = config.blobDumpNodePorts;

    if (!config.blobDumpNodeType.empty())
        dumpFilters[FILTER::BY_TYPE] = config.blobDumpNodeType;

    if (!config.blobDumpNodeName.empty())
        dumpFilters[FILTER::BY_NAME] = config.blobDumpNodeName;
}

void NodeDumper::dumpInputBlobs(const MKLDNNNodePtr& node) const {
    if (!shouldBeDumped(node, "IN"))
        return;

    auto exec_order = std::to_string(node->getExecIndex());
    std::string nodeName = node->getName();
    formatNodeName(nodeName);

    auto num_ports = node->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size();
    for (size_t i = 0; i < num_ports; i++) {
        auto prEdge = node->getParentEdgeAt(i);
        auto pr = prEdge->getParent();

        std::string file_name = NameFromType(node->getType()) + "_" + nodeName;
        if (count != -1)
            file_name += "_iter" + std::to_string(count);
        file_name += "_in" + std::to_string(i) + ".ieb";
        if (file_name.size() > 240)
            file_name = file_name.substr(file_name.size() - 240);

        auto dump_file = dumpDirName + "/#" + exec_order + "_" + file_name;
        std::cout << "Dump inputs: " << dump_file << std::endl;

        auto& desc = prEdge->getMemory().GetDesc();
        if (desc.getPrecision() == Precision::BIN)
            continue;

        // TODO [mkutakov]: rewrite BlobDumper to use MKLDNNMemory
//        BlobDumper dumper(prEdge->getBlob());
//        if (pr->ext_scales)
//            dumper.withScales(pr->ext_scales);
//
//        dump(dumper, dump_file);
    }

    dumpInternalBlobs(node);
}

void NodeDumper::dumpOutputBlobs(const MKLDNNNodePtr& node) const {
    if (!shouldBeDumped(node, "OUT"))
        return;

    auto exec_order = std::to_string(node->getExecIndex());
    std::string nodeName = node->getName();
    formatNodeName(nodeName);

    auto num_ports = node->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size();
    for (size_t i = 0; i < num_ports; i++) {
        auto childEdge = node->getChildEdgeAt(i);

        std::string file_name = NameFromType(node->getType()) + "_" + nodeName;
        if (count != -1)
            file_name += "_iter" + std::to_string(count);
        file_name += "_out" + std::to_string(i) + ".ieb";
        if (file_name.size() > 240)
            file_name = file_name.substr(file_name.size() - 240);

        auto dump_file = dumpDirName + "/#" + exec_order + "_" + file_name;
        std::cout << "Dump outputs:  " << dump_file << std::endl;

        auto& desc = childEdge->getMemory().GetDesc();
        if (desc.getPrecision() == Precision::BIN)
            continue;

        // TODO [mkutakov]: rewrite BlobDumper to use MKLDNNMemory
//        BlobDumper dumper(childEdge->getBlob());
//        if (node->ext_scales)
//            dumper.withScales(node->ext_scales);
//
//        dump(dumper, dump_file);
    }
}

void NodeDumper::dumpInternalBlobs(const MKLDNNNodePtr& node) const {
    std::string nodeName = node->getName();
    formatNodeName(nodeName);

    for (size_t i = 0; i < node->internalBlobs.size(); i++) {
        const auto& blb = node->internalBlobs[i];
        std::string file_name = NameFromType(node->getType()) + "_" + nodeName + "_blb" + std::to_string(i) + ".ieb";
        auto dump_file = dumpDirName + "/#" + std::to_string(node->getExecIndex()) + "_" + file_name;

        TensorDesc desc = blb->getTensorDesc();
        if (desc.getPrecision() == Precision::BIN)
            continue;

        BlobDumper dumper(blb);
        dump(dumper, dump_file);
    }
}

void NodeDumper::dump(const BlobDumper& bd, const std::string& file) const {
    switch (dumpFormat) {
    case FORMAT::BIN: {
        bd.dump(file);
        break;
    }
    case FORMAT::TEXT: {
        bd.dumpAsTxt(file);
        break;
    }
    default:
        IE_THROW() << "NodeDumper: Unknown dump format";
    }
}

bool NodeDumper::shouldBeDumped(const MKLDNNNodePtr& node, const std::string& portsKind) const {
    if (dumpFilters.empty())
        return false;

    if (dumpFilters.count(FILTER::BY_PORTS)) { // filter by ports configured
        if (dumpFilters.at(FILTER::BY_PORTS) != "ALL" &&
            portsKind != dumpFilters.at(FILTER::BY_PORTS))
            return false;
    }

    if (dumpFilters.count(FILTER::BY_EXEC_ID)) { // filter by exec id configured
        std::stringstream ss(dumpFilters.at(FILTER::BY_EXEC_ID));
        int id;
        bool matched = false;

        while (ss >> id) {
            if (node->getExecIndex() == id) {// exec id matches
                matched = true;
                break;
            }
        }

        if (!matched)
            return false;
    }

    if (dumpFilters.count(FILTER::BY_TYPE)) { // filter by type configured
        std::stringstream ss(dumpFilters.at(FILTER::BY_TYPE));
        std::string type;
        bool matched = false;

        while (ss >> type) {
            if (NameFromType(node->getType()) == type) {// type does not match
                matched = true;
                break;
            }
        }

        if (!matched)
            return false;
    }

    if (dumpFilters.count(FILTER::BY_NAME)) { // filter by name configured
        if (dumpFilters.at(FILTER::BY_NAME) != "*" && // to have 'single char' option for matching all the names
            !std::regex_match(node->getName(), std::regex(dumpFilters.at(FILTER::BY_NAME)))) // name does not match
            return false;
    }

    return true;
}

NodeDumper::FORMAT NodeDumper::parseDumpFormat(const std::string& format) const {
    if (format == "BIN")
        return FORMAT::BIN;
    else if (format == "TEXT")
        return FORMAT::TEXT;
    else
        IE_THROW() << "NodeDumper: Unknown dump format";
}

void NodeDumper::formatNodeName(std::string& name) const {
    std::replace(name.begin(), name.end(), '\\', '_');
    std::replace(name.begin(), name.end(), '/', '_');
    std::replace(name.begin(), name.end(), ' ', '_');
    std::replace(name.begin(), name.end(), ':', '-');
}

} // namespace MKLDNNPlugin
#endif // CPU_DEBUG_CAPS
