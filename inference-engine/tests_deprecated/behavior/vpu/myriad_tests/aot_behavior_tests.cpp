// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///  @rationale aot tests aim to test network export/import  functionality

#if defined(ENABLE_MYRIAD)

#include <behavior_test_plugin.h>
#include <mvnc.h>
#include <vpu/backend/blob_format.hpp>
#include <vpu/graph_transformer.hpp>
#include <file_utils.h>

#include "vpu_test_data.hpp"

using namespace std;
using namespace vpu;
using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace {
std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
    return obj.param.device + "_" + obj.param.input_blob_precision.name()
        + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "");
}
}

#if (defined(_WIN32) || defined(_WIN64) )
extern "C" void initialize_usb_boot();
#else
#define initialize_usb_boot()
#endif


class AOTBehaviorTests : public BehaviorPluginTest {
 public:
    WatchdogHndl_t* m_watchdogHndl = nullptr;
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;


    static std::string exported_file_name () noexcept {
        return "local_tmp.fw";
    }

    void SetUp() override {
        initialize_usb_boot();

        ASSERT_EQ(WD_ERRNO, watchdog_create(&m_watchdogHndl));
    }

    void TearDown() override {
        watchdog_destroy(m_watchdogHndl);
    }

    void dumpBlob() {
        InferenceEngine::Core core;

        CNNNetwork network = core.ReadNetwork(GetParam().model_xml_str, GetParam().weights_blob);

        ExecutableNetwork ret;
        ASSERT_NO_THROW(ret = core.LoadNetwork(network, GetParam().device, {}));

        ret.Export(exported_file_name());
    }

    void canImportBlob() {
        ASSERT_NO_THROW(importBlob()) << response.msg;
    }

    void canNotImportBlob() {
        ASSERT_THROW(importBlob(), InferenceEngine::Exception) << response.msg;
    }

    void importBlob() {
        InferenceEngine::Core{}.ImportNetwork("local_tmp.fw", GetParam().device);
    }

    void setHeaderVersion(int major, int minor) {
        FILE * f = fopen("local_tmp.fw", "r+b");
        ASSERT_NE(f, nullptr);

        ASSERT_EQ(0, fseek(f, sizeof(ElfN_Ehdr), SEEK_SET));
        mv_blob_header blobHeader;

        ASSERT_EQ(sizeof(mv_blob_header), fread(&blobHeader, 1, sizeof(mv_blob_header), f));

        ASSERT_EQ(0, fseek(f, sizeof(ElfN_Ehdr), SEEK_SET));

        blobHeader.blob_ver_major = major;
        blobHeader.blob_ver_minor = minor;

        ASSERT_EQ(sizeof(mv_blob_header), fwrite(&blobHeader, 1, sizeof(mv_blob_header), f));

        fclose(f);
    }
    std::vector<char> getBlobFileContent() {
        std::ifstream file(exported_file_name(), std::ios_base::binary);
        std::vector<char> vec;

        if (!file.eof() && !file.fail())
        {
            file.seekg(0, std::ios_base::end);
            std::streampos fileSize = file.tellg();
            vec.resize(fileSize);

            file.seekg(0, std::ios_base::beg);
            file.read(&vec[0], fileSize);
        }

        return vec;
    }

    ncDeviceHandle_t *device = nullptr;

    bool bootDevice() {
        ncStatus_t statusOpen = NC_ERROR;
        std::cout << "Opening device" << std::endl;

#ifdef  _WIN32
        const char* pathToFw = nullptr;
#else
        std::string absPathToFw = getIELibraryPath();
        const char* pathToFw = absPathToFw.c_str();
#endif //  _WIN32
        ncDeviceDescr_t deviceDesc = {};
        deviceDesc.protocol = NC_ANY_PROTOCOL;
        deviceDesc.platform = NC_ANY_PLATFORM;

        ncDeviceOpenParams_t deviceOpenParams = {};
        deviceOpenParams.watchdogHndl = m_watchdogHndl;
        deviceOpenParams.watchdogInterval = 1000;
        deviceOpenParams.customFirmwareDirectory = pathToFw;

        statusOpen = ncDeviceOpen(&device, deviceDesc, deviceOpenParams);

        if (statusOpen != NC_OK) {
            ncDeviceClose(&device, m_watchdogHndl);
            return false;
        }

        return true;
    }
};

TEST_P(AOTBehaviorTests, canImportNonModified) {
    ASSERT_NO_FATAL_FAILURE(dumpBlob());
    ASSERT_NO_FATAL_FAILURE(canImportBlob());
}

TEST_P(AOTBehaviorTests, hostSideErrorImportingIfVersionIncorrect) {

    ASSERT_NO_FATAL_FAILURE(dumpBlob());
    ASSERT_NO_FATAL_FAILURE(setHeaderVersion(vpu::BLOB_VERSION_MAJOR+1, 0));
    ASSERT_NO_FATAL_FAILURE(canNotImportBlob());
}

TEST_P(AOTBehaviorTests, canLoadGraphWithoutPlugin) {

    ASSERT_NO_FATAL_FAILURE(dumpBlob());

    auto graph = getBlobFileContent();

    ASSERT_TRUE(bootDevice());
    ncGraphHandle_t *graphHandle = nullptr;
    ASSERT_EQ(NC_OK, ncGraphCreate("aot_graph_test", &graphHandle));

    auto res = ncGraphAllocate(device, graphHandle,
                               (void*)graph.data(), graph.size(), (void*)graph.data(),
                               sizeof(ElfN_Ehdr) + sizeof(mv_blob_header));

    ncGraphDestroy(&graphHandle);
    ncDeviceClose(&device, m_watchdogHndl);

    ASSERT_EQ(NC_OK, res);
}

TEST_P(AOTBehaviorTests, deviceSideErrorImportingIfVersionIncorrect) {

    ASSERT_NO_FATAL_FAILURE(dumpBlob());
    ASSERT_NO_FATAL_FAILURE(setHeaderVersion(vpu::BLOB_VERSION_MAJOR+1, 0));

    auto graph = getBlobFileContent();

    ASSERT_TRUE(bootDevice());
    ncGraphHandle_t *graphHandle = nullptr;
    ASSERT_EQ(NC_OK, ncGraphCreate("aot_graph_test_negative", &graphHandle));

    auto res = ncGraphAllocate(device, graphHandle,
                               (void*)graph.data(), graph.size(), (void*)graph.data(),
                               sizeof(ElfN_Ehdr) + sizeof(mv_blob_header));

    ncGraphDestroy(&graphHandle);
    ncDeviceClose(&device, m_watchdogHndl);

    ASSERT_NE(NC_OK, res);
}

const BehTestParams vpuValues[] = {
    BEH_MYRIAD,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, AOTBehaviorTests, ValuesIn(vpuValues), getTestCaseName);

#endif
