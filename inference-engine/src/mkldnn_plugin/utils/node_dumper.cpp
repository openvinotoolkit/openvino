// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef CPU_DEBUG_CAPS

#include "node_dumper.h"

#include "mkldnn_node.h"
#include "utils/blob_dump.h"

#include "ie_common.h"
#include <array>
#include <regex>
#include <sstream>
#include <string>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

void NodeDumper::setup() {
    char* dumpDirEnv = getenv("OV_CPU_BLOB_DUMP_DIR");
    if (dumpDirEnv)
        dumpDirName = dumpDirEnv;

    char* dumpAsTextEnv = getenv("OV_CPU_BLOB_DUMP_AS_TEXT");
    if (dumpAsTextEnv)
        shouldDumpAsText = true;

    char* dumpInternalBlobsEnv = getenv("OV_CPU_BLOB_DUMP_INTERNAL_BLOBS");
    if (dumpInternalBlobsEnv)
        shouldDumpInternalBlobs = true;

    char* filter = getenv("OV_CPU_BLOB_DUMP_NODE_EXEC_ID");
    if (filter)
        dumpFilters[FILTER::BY_EXEC_ID] = filter;

    filter = getenv("OV_CPU_BLOB_DUMP_NODE_TYPE");
    if (filter)
        dumpFilters[FILTER::BY_TYPE] = filter;

    filter = getenv("OV_CPU_BLOB_DUMP_NODE_LAYER_TYPE");
    if (filter)
        dumpFilters[FILTER::BY_LAYER_TYPE] = filter;

    filter = getenv("OV_CPU_BLOB_DUMP_NODE_NAME");
    if (filter)
        dumpFilters[FILTER::BY_NAME] = filter;
}

void NodeDumper::dumpInputBlobs(const MKLDNNNodePtr& node) const {
    if (!shouldBeDumped(node))
        return;

    auto exec_order = std::to_string(node->getExecIndex());
    std::string nodeName = node->getName();

    std::replace(nodeName.begin(), nodeName.end(), '\\', '_');
    std::replace(nodeName.begin(), nodeName.end(), '/', '_');
    std::replace(nodeName.begin(), nodeName.end(), ' ', '_');
    std::replace(nodeName.begin(), nodeName.end(), ':', '-');

    auto num_ports = node->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size();
    for (size_t i = 0; i < num_ports; i++) {
        auto prEdge = node->getParentEdgeAt(i);
        auto pr = prEdge->getParent();

        std::string file_name = nodeName;
        if (count != -1)
            file_name += "_iter" + std::to_string(count);
        file_name += "_in" + std::to_string(i) + ".ieb";
        if (file_name.size() > 240)
            file_name = file_name.substr(file_name.size() - 240);

        auto dump_file = dumpDirName + "/#" + exec_order + "_" + file_name;
        std::cout << "Dump before: " << dump_file << std::endl;

        TensorDesc desc = prEdge->getDesc();
        if (desc.getPrecision() == Precision::BIN)
            continue;

        BlobDumper dumper(prEdge->getBlob());
        if (pr->ext_scales)
            dumper.withScales(pr->ext_scales);

        if (shouldDumpAsText)
            dumper.dumpAsTxt(dump_file);
        else
            dumper.dump(dump_file);
    }

    if (shouldDumpInternalBlobs)
        dumpInternalBlobs(node);

    return;
}

void NodeDumper::dumpOutputBlobs(const MKLDNNNodePtr& node) const {
    if (!shouldBeDumped(node))
        return;

    auto exec_order = std::to_string(node->getExecIndex());
    auto nodeName = node->getName();
    std::replace(nodeName.begin(), nodeName.end(), '\\', '_');
    std::replace(nodeName.begin(), nodeName.end(), '/', '_');
    std::replace(nodeName.begin(), nodeName.end(), ' ', '_');
    std::replace(nodeName.begin(), nodeName.end(), ':', '-');

    auto num_ports = node->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size();
    for (size_t i = 0; i < num_ports; i++) {
        auto childEdge = node->getChildEdgeAt(i);

        std::string file_name = nodeName;
        if (count != -1)
            file_name += "_iter" + std::to_string(count);
        file_name += "_out" + std::to_string(i) + ".ieb";
        if (file_name.size() > 240)
            file_name = file_name.substr(file_name.size() - 240);

        auto dump_file = dumpDirName + "/#" + exec_order + "_" + file_name;
        std::cout << "Dump after:  " << dump_file << std::endl;

        TensorDesc desc = childEdge->getDesc();
        if (desc.getPrecision() == Precision::BIN)
            continue;

        BlobDumper dumper(childEdge->getBlob());
        if (node->ext_scales)
            dumper.withScales(node->ext_scales);

        if (shouldDumpAsText)
            dumper.dumpAsTxt(dump_file);
        else
            dumper.dump(dump_file);
    }
}

void NodeDumper::dumpInternalBlobs(const MKLDNNNodePtr& node) const {
    for (size_t i = 0; i < node->internalBlobs.size(); i++) {
        const auto& blb = node->internalBlobs[i];
        auto dump_file = dumpDirName + "/#" + std::to_string(node->getExecIndex()) + "_" + node->getName() + "_blb" + std::to_string(i) + ".ieb";
        TensorDesc desc = blb->getTensorDesc();
        if (desc.getPrecision() == Precision::BIN)
            continue;
        BlobDumper dumper(blb);
        if (shouldDumpAsText)
            dumper.dumpAsTxt(dump_file);
        else
            dumper.dump(dump_file);
    }
}

bool NodeDumper::shouldBeDumped(const MKLDNNNodePtr& node) const {
    bool shouldBeDumped = false;
    const std::string& filterById = dumpFilters[FILTER::BY_EXEC_ID];

    if (!filterById.empty()) {                              // filter by exec id env set
        std::stringstream ss(filterById);
        int id;
        while (ss >> id) {
            if (node->getExecIndex() == id) // exec id matches
                shouldBeDumped = true;
        }

        if (!shouldBeDumped)
            return false;
    }

    const std::string& filterByType = dumpFilters[FILTER::BY_TYPE];

    if (!filterByType.empty()) {                           // filter by type env set
        if (NameFromType(node->getType()) != filterByType) // type does not match
            return false;
        else
            shouldBeDumped = true;
    }

    const std::string& filterByLayerType = dumpFilters[FILTER::BY_LAYER_TYPE];

    if (!filterByLayerType.empty()) {                // filter by type env set
        if (node->getTypeStr() != filterByLayerType) // layer type does not match
            return false;
        else
            shouldBeDumped = true;
    }

    const std::string& filterByName = dumpFilters[FILTER::BY_NAME];

    try {
        if (!filterByName.empty()) {                                          // filter by name env set
            if (!std::regex_match(node->getName(), std::regex(filterByName))) // name does not match
                return false;
            else
                shouldBeDumped = true;
        }
    } catch (const std::regex_error& e) {
        std::cout << e.what() << " " << e.code() << "\n";
    }

    return shouldBeDumped;
}
} // namespace MKLDNNPlugin
#endif // CPU_DEBUG_CAPS
