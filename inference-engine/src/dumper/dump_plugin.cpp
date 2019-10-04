// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dump_plugin.h"
#include "ie_plugin.hpp"
#include <description_buffer.hpp>

using namespace DumpPluginNs;
using namespace InferenceEngine;

template <typename T>
void DumpPlugin::dumpBlobTmpl(const InferenceEngine::Blob::Ptr blob, std::ofstream& file) {
    if (file.is_open()) {
        auto tblob = std::dynamic_pointer_cast<const InferenceEngine::TBlob<T>>(blob);
        if (tblob != nullptr) {
            // TODO rewrite it
            SizeVector v = tblob->dims();
            file << "dims: " << "(";
            for (auto i = v.begin(); i != v.end(); ++i)
                file << *i;
            file << ")" << std::endl;
            // file << "dims: " << "(" << tblob->dims() << ")"<< std::endl;
            for (auto it = tblob->begin(); it != tblob->end(); ++it) {
                file << std::fixed << std::setprecision(3) << *it << std::endl;
            }
        }
    }
}

void DumpPlugin::dumpBlob(const InferenceEngine::Blob::Ptr blob, std::ofstream& file) noexcept {
    switch (blob->precision()) {
        case (InferenceEngine::Precision::FP32):
            dumpBlobTmpl<float>(blob, file);
            break;
        case (InferenceEngine::Precision::FP16):
        case (InferenceEngine::Precision::Q78):
        case (InferenceEngine::Precision::I16):
            dumpBlobTmpl<short>(blob, file);
            break;
        case (InferenceEngine::Precision::U8):
            dumpBlobTmpl<char>(blob, file);
            break;
    }
}

static Version dumpPluginDescription = {
    {2, 1},  // plugin API version
    CI_BUILD_NUMBER,
    "ieDumpPlugin"  // plugin description message -
};

void DumpPlugin::GetVersion(const InferenceEngine::Version *& versionInfo) noexcept {
    versionInfo = &dumpPluginDescription;
}

std::string DumpPlugin::GetDumpDir(std::string netname) noexcept {
    static std::string dumpDir;

    if (!dumpDir.empty()) {
        return dumpDir;
    }

    const char * dump_only = getenv("DUMP_ONLY");
    if (dump_only && netname.find(dump_only) == std::string::npos) {
        dumpDir = "";
    } else {
        dumpDir = std::string(DEBUG_DUMP_PATH) + netname;
        boost::filesystem::path dir(dumpDir);

        if (!(boost::filesystem::exists(dir))) {
            boost::filesystem::create_directories(dir);
            dumpDir += "/";
        } else {
            int x = 1;
            std::string dumpDirx;

            do {
                dumpDirx = dumpDir + "_" + std::to_string(x);
                boost::filesystem::path dir2(dumpDirx);
                x++;
                if (!boost::filesystem::exists(dir2)) {
                    boost::filesystem::create_directories(dir2);
                    break;
                }
            } while (true);
            dumpDir = dumpDirx + "/";
        }
    }
    return dumpDir;
}

INFERENCE_PLUGIN_API(StatusCode) CreateDumpPlugin(IDumpPlugin*& plugin, ResponseDesc *resp) noexcept {
    try {
        plugin = new DumpPlugin();
        return OK;
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}
