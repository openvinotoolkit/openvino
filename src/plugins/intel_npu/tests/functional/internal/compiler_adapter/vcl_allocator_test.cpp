// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vcl/vcl_allocator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "model_serializer.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace intel_npu;

class VclAllocatorFuncTests : public ::testing::Test {
protected:
    void SetUp() override {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 224, 224});
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto result = std::make_shared<ov::op::v0::Result>(relu);
        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "dummy_model");

        allocator = std::make_shared<vcl_allocator_3>();
    }

    std::shared_ptr<ov::Model> model;
    std::shared_ptr<vcl_allocator_3> allocator;

    // Helper struct and function to reduce code duplication
    struct CompilerSetupState {
        std::string buildFlags;
        SerializedIR serializedIR;
        vcl_compiler_handle_t compiler = nullptr;
        vcl_log_handle_t logHandle = nullptr;
        vcl_executable_desc_t desc = {};

        ~CompilerSetupState() {
            if (compiler != nullptr && VCLApi::getInstance()) {
                VCLApi::getInstance()->vclCompilerDestroy(compiler);
            }
        }
    };

    CompilerSetupState createCompilerAndDescriptor() {
        CompilerSetupState state;

        state.buildFlags = intel_npu::compiler_utils::serializeIOInfo(model, true);
        state.buildFlags += " --config NPU_PLATFORM=\"3720\" NPU_COMPILATION_MODE_PARAMS=\"optimization-level=0\"";

        if (auto vclApi = VCLApi::getInstance()) {
            vcl_version_info_t vclVersion = {};
            vcl_version_info_t vclProfilingVersion = {};
            vclApi->vclGetVersion(&vclVersion, &vclProfilingVersion);

            vcl_compiler_desc_t compilerDesc = {};
            compilerDesc.version = vclVersion;
            compilerDesc.debugLevel = static_cast<__vcl_log_level_t>(3);

            uint32_t defaultTileCount = std::numeric_limits<uint32_t>::max();
            if (vclVersion.major == 7 && vclVersion.minor < 6) {
                defaultTileCount = std::numeric_limits<uint16_t>::max();
            }

            vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t),
                                            0x00,
                                            std::numeric_limits<uint16_t>::max(),
                                            defaultTileCount};

            vclApi->vclCompilerCreate(&compilerDesc, &deviceDesc, &state.compiler, &state.logHandle);
        }
        if (state.compiler == nullptr) {
            ADD_FAILURE() << "vclCompilerCreate failed";
            return state;
        }

        ze_graph_compiler_version_info_t vclVersionInfo = {0, 0};
        if (auto vclApi = VCLApi::getInstance()) {
            vcl_compiler_properties_t compilerProp = {};
            vclApi->vclCompilerGetProperties(state.compiler, &compilerProp);
            vclVersionInfo.major = compilerProp.version.major;
            vclVersionInfo.minor = compilerProp.version.minor;
        }

        auto isOptionValueSupportedByCompiler = [](const std::string&, const std::optional<std::string>&) {
            return true;
        };

        state.serializedIR = intel_npu::compiler_utils::serializeIR(model,
                                                                    vclVersionInfo,
                                                                    10,
                                                                    MODEL_SERIALIZER_VERSION::defaultValue(),
                                                                    isOptionValueSupportedByCompiler);

        state.buildFlags += " NPU_MODEL_SERIALIZER_VERSION=\"" +
                            intel_npu::MODEL_SERIALIZER_VERSION::toString(state.serializedIR.serializerVersion) + "\"";

        state.desc = {state.serializedIR.buffer.get(),
                      state.serializedIR.size,
                      state.buildFlags.c_str(),
                      state.buildFlags.size()};

        return state;
    }
};

TEST_F(VclAllocatorFuncTests, CheckVclAllocatedExecutableCreate2) {
    uint8_t* blobBuffer = nullptr;
    uint64_t blobSize = 0;

    auto setup = createCompilerAndDescriptor();

    auto result =
        intel_npu::vclAllocatedExecutableCreate2(setup.compiler, setup.desc, allocator.get(), &blobBuffer, &blobSize);

    EXPECT_EQ(result, VCL_RESULT_SUCCESS);
    EXPECT_NE(blobBuffer, nullptr);

    auto it = std::find_if(allocator->m_info.begin(),
                           allocator->m_info.end(),
                           [blobBuffer](const std::pair<uint8_t*, size_t>& item) {
                               return item.first == blobBuffer;
                           });

    EXPECT_NE(it, allocator->m_info.end());
    if (it != allocator->m_info.end()) {
        size_t alignedBlobSize = it->second;
        ov::Tensor alignedBlob = make_tensor_from_aligned_addr(blobBuffer, alignedBlobSize, allocator);

        EXPECT_EQ(alignedBlob.data(), blobBuffer);
        EXPECT_EQ(alignedBlob.get_byte_size(), alignedBlobSize);

        allocator->m_info.erase(it);
    }
}
