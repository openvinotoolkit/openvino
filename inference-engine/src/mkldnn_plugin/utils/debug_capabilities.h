// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#include <map>
#include <string>
#include <vector>

#define ENABLE_CPU_DEBUG_CAP(_x) _x;

namespace MKLDNNPlugin {
namespace DebugCaps {

class Config {
public:
    Config() {
        readParam(blobDumpDir, "OV_CPU_BLOB_DUMP_DIR");
        readParam(blobDumpFormat, "OV_CPU_BLOB_DUMP_FORMAT");
        readParam(blobDumpNodeExecId, "OV_CPU_BLOB_DUMP_NODE_EXEC_ID");
        readParam(blobDumpNodeType, "OV_CPU_BLOB_DUMP_NODE_TYPE");
        readParam(blobDumpNodeName, "OV_CPU_BLOB_DUMP_NODE_NAME");
        readParam(execGraphPath, "OV_CPU_EXEC_GRAPH_PATH");
    }

    std::string blobDumpDir;
    std::string blobDumpFormat;
    std::string blobDumpNodeExecId;
    std::string blobDumpNodeType;
    std::string blobDumpNodeName;
    std::string execGraphPath;

private:
    void readParam(std::string& param, const char* envVar) {
        if (const char* envValue = std::getenv(envVar))
            param = envValue;
    }
};

} // namespace DebugCaps
} // namespace MKLDNNPlugin

#else // !CPU_DEBUG_CAPS
#define ENABLE_CPU_DEBUG_CAP(_x)
#endif // CPU_DEBUG_CAPS
