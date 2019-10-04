// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/ie_irelease.hpp"
#include "ie_version.hpp"
#include "ie_common.h"
#include "ie_blob.h"
#include <string>

class IDumpPlugin : public InferenceEngine::details::IRelease {
public:
    /**
     * @brief return plugin's version information
     * @param versionInfo pointer to version info, will be set by plugin
     */
    virtual void GetVersion(const InferenceEngine::Version *& versionInfo) noexcept = 0;

    virtual void dumpBlob(const InferenceEngine::Blob::Ptr blob, std::ofstream& file) noexcept = 0;

    virtual std::string GetDumpDir(std::string networkName) noexcept = 0;
};

#ifdef DEBUG_DUMP
#include "ie_dump_plugin_ptr.hpp"
static DumpPluginPtr dumper("libieDumper.so");
#define DUMP_BLOB(blob, file) \
    dumper->dumpBlob(blob, file);
#else
#define DUMP_BLOB(blob, file)
#endif  // DEBUG_DUMP
