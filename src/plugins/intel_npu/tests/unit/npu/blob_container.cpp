// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/subgraph_builders/2_input_subtract.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_constants.hpp"
#include "dev/core_impl.hpp"
#include "intel_npu/common/icompiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/file_path.hpp"

using namespace intel_npu;

using BlobContainerUnitTests = ::testing::Test;

namespace {
const char* dummyBlobHeader = "blobwillstartafterspace correctblob!";
const char* testCacheDir = "blob_container_test_cache_dir";
const char* testFileName = "blob_container_test.blob";

}  // namespace

TEST_F(BlobContainerUnitTests, isBlobContainerCorrectlyPickedForCacheEnabled) {
    auto core = std::make_shared<ov::CoreImpl>();
    core->register_compile_time_plugins();
    auto model = ov::test::utils::make_2_input_subtract();

    {
        auto compiledModel = core->compile_model(model,
                                                 ov::test::utils::DEVICE_NPU,
                                                 {ov::cache_dir(testCacheDir), ov::enable_profiling(true)});
        auto inferRequest = compiledModel->create_infer_request();
        inferRequest->infer();
        OV_ASSERT_NO_THROW(auto profilingInfo = inferRequest->get_profiling_info());

        auto* compiledModelPtr = dynamic_cast<intel_npu::ICompiledModel*>(compiledModel._ptr.get());
        OPENVINO_ASSERT(compiledModelPtr != nullptr);
        const auto& blobContainer = compiledModelPtr->get_graph()->get_blob_container();
        auto* blobContainerAlignedBufferPtr =
            dynamic_cast<const intel_npu::BlobContainerAlignedBuffer*>(&blobContainer);
        OPENVINO_ASSERT(blobContainerAlignedBufferPtr == nullptr,
                        "Blob after compilation should not be memory mapped!");
    }

    {
        auto compiledModel = core->compile_model(model,
                                                 ov::test::utils::DEVICE_NPU,
                                                 {ov::cache_dir(testCacheDir), ov::enable_profiling(true)});
        auto inferRequest = compiledModel->create_infer_request();
        inferRequest->infer();
        OV_ASSERT_NO_THROW(auto profilingInfo = inferRequest->get_profiling_info());
        auto outputFile =
            std::ofstream(std::filesystem::path(testCacheDir) / testFileName, std::ios::out | std::ios::binary);
        OV_ASSERT_NO_THROW(compiledModel->export_model(outputFile));

        auto* compiledModelPtr = dynamic_cast<intel_npu::ICompiledModel*>(compiledModel._ptr.get());
        OPENVINO_ASSERT(compiledModelPtr != nullptr);
        const auto& blobContainer = compiledModelPtr->get_graph()->get_blob_container();
        auto* blobContainerAlignedBufferPtr =
            dynamic_cast<const intel_npu::BlobContainerAlignedBuffer*>(&blobContainer);
        OPENVINO_ASSERT(blobContainerAlignedBufferPtr != nullptr, "Cached blob should be memory mapped!");
    }
    ov::test::utils::removeDir(testCacheDir);
}

TEST_F(BlobContainerUnitTests, isBlobContainerCorrectlyPickedForFStream) {
    auto core = std::make_shared<ov::CoreImpl>();
    core->register_compile_time_plugins();
    auto model = ov::test::utils::make_2_input_subtract();

    {
        auto compiledModel = core->compile_model(model, ov::test::utils::DEVICE_NPU, {ov::enable_profiling(true)});
        auto inferRequest = compiledModel->create_infer_request();
        inferRequest->infer();
        OV_ASSERT_NO_THROW(auto profilingInfo = inferRequest->get_profiling_info());
        auto outputFile = std::ofstream(testFileName, std::ios::out | std::ios::binary);
        OV_ASSERT_NO_THROW(compiledModel->export_model(outputFile));
    }

    {
        auto inputFile = std::ifstream(testFileName, std::ios::in | std::ios::binary);
        auto compiledModel = core->import_model(inputFile, ov::test::utils::DEVICE_NPU, {ov::enable_profiling(true)});
        inputFile.close();
        auto inferRequest = compiledModel->create_infer_request();
        inferRequest->infer();
        OV_ASSERT_NO_THROW(auto profilingInfo = inferRequest->get_profiling_info());
        auto outputFile = std::ofstream(testFileName, std::ios::out | std::ios::binary);
        OV_ASSERT_NO_THROW(compiledModel->export_model(outputFile));

        auto* compiledModelPtr = dynamic_cast<intel_npu::ICompiledModel*>(compiledModel._ptr.get());
        OPENVINO_ASSERT(compiledModelPtr != nullptr);
        const auto& blobContainer = compiledModelPtr->get_graph()->get_blob_container();
        auto* blobContainerAlignedBufferPtr =
            dynamic_cast<const intel_npu::BlobContainerAlignedBuffer*>(&blobContainer);
        OPENVINO_ASSERT(blobContainerAlignedBufferPtr == nullptr, "Cannot have memory mapped blob for std::fstream!");
    }
    ov::test::utils::removeFile(testFileName);
}

TEST_F(BlobContainerUnitTests, isBlobContainerCorrectlyPickedForSStream) {
    auto core = std::make_shared<ov::CoreImpl>();
    core->register_compile_time_plugins();
    auto model = ov::test::utils::make_2_input_subtract();
    std::stringstream blobStream;

    {
        auto compiledModel = core->compile_model(model, ov::test::utils::DEVICE_NPU, {ov::enable_profiling(true)});
        auto inferRequest = compiledModel->create_infer_request();
        inferRequest->infer();
        OV_ASSERT_NO_THROW(auto profilingInfo = inferRequest->get_profiling_info());
        OV_ASSERT_NO_THROW(compiledModel->export_model(blobStream));
    }

    {
        auto compiledModel = core->import_model(blobStream, ov::test::utils::DEVICE_NPU, {ov::enable_profiling(true)});
        blobStream = std::stringstream();
        auto inferRequest = compiledModel->create_infer_request();
        inferRequest->infer();
        OV_ASSERT_NO_THROW(auto profilingInfo = inferRequest->get_profiling_info());
        OV_ASSERT_NO_THROW(compiledModel->export_model(blobStream));

        auto* compiledModelPtr = dynamic_cast<intel_npu::ICompiledModel*>(compiledModel._ptr.get());
        OPENVINO_ASSERT(compiledModelPtr != nullptr);
        const auto& blobContainer = compiledModelPtr->get_graph()->get_blob_container();
        auto* blobContainerAlignedBufferPtr =
            dynamic_cast<const intel_npu::BlobContainerAlignedBuffer*>(&blobContainer);
        OPENVINO_ASSERT(blobContainerAlignedBufferPtr == nullptr,
                        "Cannot have memory mapped blob for std::stringstream!");
    }
}

TEST_F(BlobContainerUnitTests, isBlobHeaderHandledCorrectly) {
    auto core = std::make_shared<ov::CoreImpl>();
    core->register_compile_time_plugins();
    auto model = ov::test::utils::make_2_input_subtract();
    std::stringstream blobStream;
    blobStream << dummyBlobHeader;

    {
        auto compiledModel = core->compile_model(model, ov::test::utils::DEVICE_NPU, {ov::enable_profiling(true)});
        auto inferRequest = compiledModel->create_infer_request();
        inferRequest->infer();
        OV_ASSERT_NO_THROW(auto profilingInfo = inferRequest->get_profiling_info());
        auto outputFile = std::ofstream(testFileName, std::ios::out | std::ios::binary);
        outputFile << dummyBlobHeader;
        OV_ASSERT_NO_THROW(compiledModel->export_model(outputFile));
        OV_ASSERT_NO_THROW(compiledModel->export_model(blobStream));
    }

    {
        std::string parseDummyHeader;
        std::string blob;
        blobStream >> parseDummyHeader;

        EXPECT_THAT(parseDummyHeader, testing::HasSubstr("blobwillstartafterspace"));
        auto compiledModel =
            core->import_model(blobStream, ov::test::utils::DEVICE_NPU, {ov::intel_npu::defer_weights_load(true)});
        blobStream = {};

        auto* compiledModelPtr = dynamic_cast<intel_npu::ICompiledModel*>(compiledModel._ptr.get());
        OPENVINO_ASSERT(compiledModelPtr != nullptr);
        const auto& blobContainer = compiledModelPtr->get_graph()->get_blob_container();
        blob.assign(reinterpret_cast<const char*>(blobContainer.get_ptr()), blobContainer.size());
        EXPECT_THAT(blob, testing::HasSubstr("correctblob!"));
    }

    {
        std::string parseDummyHeader;
        std::string blob;
        auto inputFile = std::ifstream(testFileName, std::ios::in | std::ios::binary);
        blobStream >> parseDummyHeader;

        EXPECT_THAT(parseDummyHeader, testing::HasSubstr("blobwillstartafterspace"));
        auto compiledModel =
            core->import_model(blobStream, ov::test::utils::DEVICE_NPU, {ov::intel_npu::defer_weights_load(true)});

        auto* compiledModelPtr = dynamic_cast<intel_npu::ICompiledModel*>(compiledModel._ptr.get());
        OPENVINO_ASSERT(compiledModelPtr != nullptr);
        const auto& blobContainer = compiledModelPtr->get_graph()->get_blob_container();
        blob.assign(reinterpret_cast<const char*>(blobContainer.get_ptr()), blobContainer.size());
        EXPECT_THAT(blob, testing::HasSubstr("correctblob!"));
    }

    ov::test::utils::removeFile(testFileName);
}
