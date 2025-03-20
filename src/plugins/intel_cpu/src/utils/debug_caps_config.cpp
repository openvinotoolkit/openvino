// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef CPU_DEBUG_CAPS

#    include "debug_caps_config.h"

#    include <string>

namespace ov::intel_cpu {

void DebugCapsConfig::readProperties() {
    auto readEnv = [](const char* envVar) {
        const char* env = std::getenv(envVar);
        if (env && *env) {  // set and non-empty
            return env;
        }

        return static_cast<const char*>(nullptr);
    };

    auto parseDumpFormat = [](const std::string& format) {
        if (format == "BIN") {
            return FORMAT::BIN;
        }
        if (format == "TEXT") {
            return FORMAT::TEXT;
        }
        OPENVINO_THROW("readDebugCapsProperties: Unknown dump format");
    };

    if (auto envVarValue = readEnv("OV_CPU_EXEC_GRAPH_PATH")) {
        execGraphPath = envVarValue;
    }

    if (auto envVarValue = readEnv("OV_CPU_VERBOSE")) {
        verbose = envVarValue;
    }

    if (auto envVarValue = readEnv("OV_CPU_BLOB_DUMP_DIR")) {
        blobDumpDir = envVarValue;
    }

    if (auto envVarValue = readEnv("OV_CPU_BLOB_DUMP_FORMAT")) {
        blobDumpFormat = parseDumpFormat(envVarValue);
    }

    if (auto envVarValue = readEnv("OV_CPU_BLOB_DUMP_NODE_EXEC_ID")) {
        blobDumpFilters[FILTER::BY_EXEC_ID] = envVarValue;
    }

    if (auto envVarValue = readEnv("OV_CPU_BLOB_DUMP_NODE_PORTS")) {
        blobDumpFilters[FILTER::BY_PORTS] = envVarValue;
    }

    if (auto envVarValue = readEnv("OV_CPU_BLOB_DUMP_NODE_TYPE")) {
        blobDumpFilters[FILTER::BY_TYPE] = envVarValue;
    }

    if (auto envVarValue = readEnv("OV_CPU_BLOB_DUMP_NODE_NAME")) {
        blobDumpFilters[FILTER::BY_NAME] = envVarValue;
    }

    if (auto envVarValue = readEnv("OV_CPU_DISABLE")) {
        disable.parseAndSet(envVarValue);
    }

    if (auto envVarValue = readEnv("OV_CPU_DUMP_IR")) {
        dumpIR.parseAndSet(envVarValue);
    }

    if (auto envVarValue = readEnv("OV_CPU_SUMMARY_PERF")) {
        summaryPerf = envVarValue;
    }

    if (auto envVarValue = readEnv("OV_CPU_AVERAGE_COUNTERS")) {
        averageCountersPath = envVarValue;
    }

    if (auto envVarValue = readEnv("OV_CPU_MEMORY_STATISTICS_PATH")) {
        memoryStatisticsDumpPath = envVarValue;
    }
}

}  // namespace ov::intel_cpu

#endif  // CPU_DEBUG_CAPS
