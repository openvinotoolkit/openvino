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
    bool shouldBeDumped(const MKLDNNNodePtr &node) const;

    enum class DUMP_FORMAT {
        BIN,
        TEXT,
    };

    DUMP_FORMAT parseDumpFormat(const std::string& format) const;
    void formatNodeName(std::string& name) const;

    DUMP_FORMAT dumpFormat;
    std::string dumpDirName;
    int count;

    enum FILTER {
        BY_EXEC_ID,
        BY_TYPE,
        BY_NAME,
        COUNT,
    };

    std::unordered_map<FILTER, std::string> dumpFilters;
};
} // namespace MKLDNNPlugin
#endif // CPU_DEBUG_CAPS
