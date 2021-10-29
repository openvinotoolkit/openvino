// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#include <map>
#include <string>
#include <vector>


#define CPU_DEBUG_CAP_ENABLE(_x) _x;
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) true

namespace MKLDNNPlugin {
namespace DebugCaps {

class Config {
public:
    Config() {
        readParam(blobDumpDir, "OV_CPU_BLOB_DUMP_DIR");
        readParam(blobDumpFormat, "OV_CPU_BLOB_DUMP_FORMAT");
        readParam(blobDumpNodeExecId, "OV_CPU_BLOB_DUMP_NODE_EXEC_ID");
        readParam(blobDumpNodePorts, "OV_CPU_BLOB_DUMP_NODE_PORTS");
        readParam(blobDumpNodeType, "OV_CPU_BLOB_DUMP_NODE_TYPE");
        readParam(blobDumpNodeName, "OV_CPU_BLOB_DUMP_NODE_NAME");
        readParam(execGraphPath, "OV_CPU_EXEC_GRAPH_PATH");
        readParam(verbose, "OV_CPU_VERBOSE");
    }

    std::string blobDumpDir;
    std::string blobDumpFormat;
    std::string blobDumpNodeExecId;
    std::string blobDumpNodePorts;
    std::string blobDumpNodeType;
    std::string blobDumpNodeName;
    std::string execGraphPath;
    std::string verbose;

private:
    static void readParam(std::string& param, const char* envVar) {
        if (const char* envValue = std::getenv(envVar))
            param = envValue;
    }
};

} // namespace DebugCaps
} // namespace MKLDNNPlugin

#else // !CPU_DEBUG_CAPS
#define CPU_DEBUG_CAP_ENABLE(_x)
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) x

#endif // CPU_DEBUG_CAPS
