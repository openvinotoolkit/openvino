// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_dump_plugin.hpp"
#include <boost/filesystem.hpp>
#include <iomanip>
#include <fstream>
#include <string>

#define DEBUG_DUMP_PATH "dump/"

namespace DumpPluginNs {

using namespace boost::filesystem;

class DumpPlugin : public IDumpPlugin {
public:
    DumpPlugin() {}

    virtual ~DumpPlugin() {}

    void GetVersion(const InferenceEngine::Version *& versionInfo) noexcept override;

    void Release() noexcept override {
        delete this;
    }

    void dumpBlob(const InferenceEngine::Blob::Ptr blob, std::ofstream& file) noexcept override;

    std::string GetDumpDir(std::string networkName) noexcept override;

private:
    template <typename T>
    void dumpBlobTmpl(const InferenceEngine::Blob::Ptr blob, std::ofstream& file);
};

}  // namespace DumpPluginNs
