// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef CPU_DEBUG_CAPS
#pragma once

#include "mkldnn_node.h"
#include "utils/blob_dump.h"
#include "utils/debug_capabilities.h"

#include <unordered_map>
#include <string>

namespace MKLDNNPlugin {

/**
 * Blobs are not dumped by default
 * Blobs are dumped if node matches all specified env filters
 *
 * To dump blobs from all the nodes use the following filter:
 *
 * OV_CPU_BLOB_DUMP_NODE_NAME=.+
 */
class NodeDumper {
public:
    NodeDumper(const DebugCaps::Config& config, const int _count);

    void dumpInputBlobs(const MKLDNNNodePtr &node) const;
    void dumpOutputBlobs(const MKLDNNNodePtr &node) const;

private:
    void dumpInternalBlobs(const MKLDNNNodePtr& node) const;
    void dump(const BlobDumper& bd, const std::string& file) const;
    bool shouldBeDumped(const MKLDNNNodePtr &node, const std::string& portsKind) const;

    enum class FORMAT {
        BIN,
        TEXT,
    };

    FORMAT parseDumpFormat(const std::string& format) const;
    void formatNodeName(std::string& name) const;

    FORMAT dumpFormat;
    std::string dumpDirName;
    int count;

    enum FILTER {
        BY_PORTS,
        BY_EXEC_ID,
        BY_TYPE,
        BY_NAME,
    };

    // std::hash<int> is necessary for Ubuntu-16.04 (gcc-5.4 and defect in C++11 standart)
    std::unordered_map<FILTER, std::string, std::hash<int>> dumpFilters;
};
} // namespace MKLDNNPlugin
#endif // CPU_DEBUG_CAPS
