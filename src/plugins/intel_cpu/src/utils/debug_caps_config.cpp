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
        if ((env != nullptr) && (*env != 0)) {  // set and non-empty
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

    const char* envVarValue = nullptr;

    if ((envVarValue = readEnv("OV_CPU_EXEC_GRAPH_PATH")) != nullptr) {
        execGraphPath = envVarValue;
    }

    if ((envVarValue = readEnv("OV_CPU_VERBOSE")) != nullptr) {
        verbose = envVarValue;
    }

    if ((envVarValue = readEnv("OV_CPU_BLOB_DUMP_DIR")) != nullptr) {
        blobDumpDir = envVarValue;
    }

    if ((envVarValue = readEnv("OV_CPU_BLOB_DUMP_FORMAT")) != nullptr) {
        blobDumpFormat = parseDumpFormat(envVarValue);
    }

    if ((envVarValue = readEnv("OV_CPU_BLOB_DUMP_NODE_EXEC_ID")) != nullptr) {
        blobDumpFilters[FILTER::BY_EXEC_ID] = envVarValue;
    }

    if ((envVarValue = readEnv("OV_CPU_BLOB_DUMP_NODE_PORTS")) != nullptr) {
        blobDumpFilters[FILTER::BY_PORTS] = envVarValue;
    }

    if ((envVarValue = readEnv("OV_CPU_BLOB_DUMP_NODE_TYPE")) != nullptr) {
        blobDumpFilters[FILTER::BY_TYPE] = envVarValue;
    }

    if ((envVarValue = readEnv("OV_CPU_BLOB_DUMP_NODE_NAME")) != nullptr) {
        blobDumpFilters[FILTER::BY_NAME] = envVarValue;
    }

    if ((envVarValue = readEnv("OV_CPU_DISABLE")) != nullptr) {
        disable.parseAndSet(envVarValue);
    }

    if ((envVarValue = readEnv("OV_CPU_DUMP_IR")) != nullptr) {
        dumpIR.parseAndSet(envVarValue);
    }

    if ((envVarValue = readEnv("OV_CPU_SUMMARY_PERF")) != nullptr) {
        summaryPerf = envVarValue;
    }

    if ((envVarValue = readEnv("OV_CPU_AVERAGE_COUNTERS")) != nullptr) {
        averageCountersPath = envVarValue;
    }
}

}  // namespace ov::intel_cpu

#endif  // CPU_DEBUG_CAPS
