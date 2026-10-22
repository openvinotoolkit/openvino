// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <array>
#include <common_test_utils/test_common.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include "common/npu_test_env_cfg.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_mem.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "vm_runtime/npu_vm_runtime.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/util/common_util.hpp>
#include <openvino/util/shared_object.hpp>
#include <sstream>
#include <tuple>

struct npu_vm_runtime_fntbl {
    using npu_vm_runtime_get_api_version_t = npu_vm_runtime_result_t(npu_vm_runtime_version_t* pVersion);
    using npu_vm_runtime_create_t = npu_vm_runtime_result_t(const npu_vm_runtime_blob_desc_t* desc,
                                                            npu_vm_runtime_handle_t* phRuntime,
                                                            npu_vm_runtime_properties_t* pProperties);
    using npu_vm_runtime_destroy_t = npu_vm_runtime_result_t(npu_vm_runtime_handle_t hRuntime);
    using npu_vm_runtime_get_metadata_t =
            npu_vm_runtime_result_t(npu_vm_runtime_handle_t hRuntime, uint32_t argIndex,
                                    ze_graph_argument_properties_3_t* pGraphArgumentProperties,
                                    ze_graph_argument_metadata_t* pGraphArgumentMetadata, int64_t* upperBound);
    using npu_vm_runtime_execute_t = npu_vm_runtime_result_t(npu_vm_runtime_handle_t hRuntime,
                                                             npu_vm_runtime_execute_params_t* pParams);
    using npu_vm_runtime_create_execution_context_t = npu_vm_runtime_result_t(
            npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_execution_context_handle_t* phExecutionHandle);
    using npu_vm_runtime_destroy_execution_context_t =
            npu_vm_runtime_result_t(npu_vm_runtime_execution_context_handle_t phExecutionHandle);
    using npu_vm_runtime_update_mutable_commandlist_t =
            npu_vm_runtime_result_t(npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_execute_params_t* pParams,
                                    uint64_t* argIndexArray, uint64_t argIndexArraySize);
    using npu_vm_runtime_predict_output_shape_t = npu_vm_runtime_result_t(
            npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_predict_output_shape_params_t* pParams);

    using npu_vm_runtime_create_mem_ref_t = npu_vm_runtime_result_t(int64_t dimsCount,
                                                                    npu_vm_runtime_mem_ref_handle_t* phMemRef);

    using npu_vm_runtime_destroy_mem_ref_t = npu_vm_runtime_result_t(npu_vm_runtime_mem_ref_handle_t hMemRef);

    using npu_vm_runtime_set_mem_ref_t = npu_vm_runtime_result_t(npu_vm_runtime_mem_ref_handle_t hMemRef,
                                                                 const void* basePtr, const void* data, int64_t offset,
                                                                 int64_t* pSizes, int64_t* pStrides, int64_t dimsCount);

    using npu_vm_runtime_parse_mem_ref_t = npu_vm_runtime_result_t(npu_vm_runtime_mem_ref_handle_t hMemRef,
                                                                   const void** pBasePtr, const void** pData,
                                                                   int64_t* pOffset, int64_t* pSizes, int64_t* pStrides,
                                                                   int64_t* pDimsCount);

    using npu_vm_runtime_predict_output_shape2_t = npu_vm_runtime_result_t(
            npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_predict_output_shape_params_t2* pParams);

    using npu_vm_runtime_create2_t = npu_vm_runtime_result_t(const npu_vm_runtime_blob_desc_t* desc,
                                                             const npu_vm_runtime_config_desc_t* pConfig,
                                                             npu_vm_runtime_handle_t* phRuntime,
                                                             npu_vm_runtime_properties_t* pProperties);
    using npu_vm_runtime_create_execution_context2_t = npu_vm_runtime_result_t(
            npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_create_execution_context_params_t* pParams,
            npu_vm_runtime_execution_context_handle_t* phExecutionHandle);
    using npu_vm_runtime_execute2_t = npu_vm_runtime_result_t(npu_vm_runtime_handle_t hRuntime,
                                                              npu_vm_runtime_execute_params2_t* pParams);
    using npu_vm_runtime_host_sync_t = npu_vm_runtime_result_t(npu_vm_runtime_handle_t hRuntime,
                                                               npu_vm_runtime_execute_params2_t* pParams);

    std::function<npu_vm_runtime_get_api_version_t> npuVMRuntimeGetAPIVersion = nullptr;
    std::function<npu_vm_runtime_create_t> npuVMRuntimeCreate = nullptr;
    std::function<npu_vm_runtime_destroy_t> npuVMRuntimeDestroy = nullptr;
    std::function<npu_vm_runtime_get_metadata_t> npuVMRuntimeGetMetadata = nullptr;
    std::function<npu_vm_runtime_execute_t> npuVMRuntimeExecute = nullptr;
    std::function<npu_vm_runtime_create_execution_context_t> npuVMRuntimeCreateExecutionContext = nullptr;
    std::function<npu_vm_runtime_destroy_execution_context_t> npuVMRuntimeDestroyExecutionContext = nullptr;
    std::function<npu_vm_runtime_update_mutable_commandlist_t> npuVMRuntimeUpdateMutableCommandList = nullptr;
    std::function<npu_vm_runtime_predict_output_shape_t> npuVMRuntimePredictOutputShape = nullptr;
    std::function<npu_vm_runtime_create_mem_ref_t> npuVMRuntimeCreateMemRef = nullptr;
    std::function<npu_vm_runtime_destroy_mem_ref_t> npuVMRuntimeDestroyMemRef = nullptr;
    std::function<npu_vm_runtime_set_mem_ref_t> npuVMRuntimeSetMemRef = nullptr;
    std::function<npu_vm_runtime_parse_mem_ref_t> npuVMRuntimeParseMemRef = nullptr;
    std::function<npu_vm_runtime_predict_output_shape2_t> npuVMRuntimePredictOutputShape2 = nullptr;
    std::function<npu_vm_runtime_create2_t> npuVMRuntimeCreate2 = nullptr;
    std::function<npu_vm_runtime_create_execution_context2_t> npuVMRuntimeCreateExecutionContext2 = nullptr;
    std::function<npu_vm_runtime_execute2_t> npuVMRuntimeExecute2 = nullptr;
    std::function<npu_vm_runtime_host_sync_t> npuVMRuntimeHostSync = nullptr;

    void init(std::shared_ptr<void> lib, const std::string&) {
        npuVMRuntimeGetAPIVersion = reinterpret_cast<npu_vm_runtime_get_api_version_t*>(
                ov::util::get_symbol(lib, "npuVMRuntimeGetAPIVersion"));
        npuVMRuntimeCreate =
                reinterpret_cast<npu_vm_runtime_create_t*>(ov::util::get_symbol(lib, "npuVMRuntimeCreate"));
        npuVMRuntimeDestroy =
                reinterpret_cast<npu_vm_runtime_destroy_t*>(ov::util::get_symbol(lib, "npuVMRuntimeDestroy"));
        npuVMRuntimeGetMetadata =
                reinterpret_cast<npu_vm_runtime_get_metadata_t*>(ov::util::get_symbol(lib, "npuVMRuntimeGetMetadata"));
        npuVMRuntimeExecute =
                reinterpret_cast<npu_vm_runtime_execute_t*>(ov::util::get_symbol(lib, "npuVMRuntimeExecute"));

        npuVMRuntimeCreateExecutionContext = reinterpret_cast<npu_vm_runtime_create_execution_context_t*>(
                ov::util::get_symbol(lib, "npuVMRuntimeCreateExecutionContext"));
        npuVMRuntimeDestroyExecutionContext = reinterpret_cast<npu_vm_runtime_destroy_execution_context_t*>(
                ov::util::get_symbol(lib, "npuVMRuntimeDestroyExecutionContext"));
        npuVMRuntimeUpdateMutableCommandList = reinterpret_cast<npu_vm_runtime_update_mutable_commandlist_t*>(
                ov::util::get_symbol(lib, "npuVMRuntimeUpdateMutableCommandList"));

        npuVMRuntimePredictOutputShape = reinterpret_cast<npu_vm_runtime_predict_output_shape_t*>(
                ov::util::get_symbol(lib, "npuVMRuntimePredictOutputShape"));
        npuVMRuntimeCreateMemRef = reinterpret_cast<npu_vm_runtime_create_mem_ref_t*>(
                ov::util::get_symbol(lib, "npuVMRuntimeCreateMemRef"));
        npuVMRuntimeDestroyMemRef = reinterpret_cast<npu_vm_runtime_destroy_mem_ref_t*>(
                ov::util::get_symbol(lib, "npuVMRuntimeDestroyMemRef"));
        npuVMRuntimeSetMemRef =
                reinterpret_cast<npu_vm_runtime_set_mem_ref_t*>(ov::util::get_symbol(lib, "npuVMRuntimeSetMemRef"));
        npuVMRuntimeParseMemRef =
                reinterpret_cast<npu_vm_runtime_parse_mem_ref_t*>(ov::util::get_symbol(lib, "npuVMRuntimeParseMemRef"));
        npuVMRuntimePredictOutputShape2 = reinterpret_cast<npu_vm_runtime_predict_output_shape2_t*>(
                ov::util::get_symbol(lib, "npuVMRuntimePredictOutputShape2"));

        npuVMRuntimeCreate2 =
                reinterpret_cast<npu_vm_runtime_create2_t*>(ov::util::get_symbol(lib, "npuVMRuntimeCreate2"));
        npuVMRuntimeCreateExecutionContext2 = reinterpret_cast<npu_vm_runtime_create_execution_context2_t*>(
                ov::util::get_symbol(lib, "npuVMRuntimeCreateExecutionContext2"));
        npuVMRuntimeExecute2 =
                reinterpret_cast<npu_vm_runtime_execute2_t*>(ov::util::get_symbol(lib, "npuVMRuntimeExecute2"));
        npuVMRuntimeHostSync =
                reinterpret_cast<npu_vm_runtime_execute2_t*>(ov::util::get_symbol(lib, "npuVMRuntimeHostSync"));
    }
};

std::shared_ptr<ov::Model> createMaxPoolModel() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                         ov::PartialShape{1, 16, 720, ov::Dimension(10, 1280)});
    input->set_friendly_name("input1");

    auto maxpool = std::make_shared<ov::op::v1::MaxPool>(input, ov::Strides{1, 1}, ov::Shape{0, 0}, ov::Shape{0, 0},
                                                         ov::Shape{1, 1}, ov::op::RoundingType::FLOOR,
                                                         ov::op::PadType::EXPLICIT);
    maxpool->set_friendly_name("MaxPool_2");

    auto result = std::make_shared<ov::op::v0::Result>(maxpool);
    result->set_friendly_name("output");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "MaxPool");
}

namespace {

std::string formatCheckedPaths(const std::vector<std::filesystem::path>& paths) {
    std::ostringstream oss;
    for (size_t i = 0; i < paths.size(); ++i) {
        if (i != 0) {
            oss << "; ";
        }
        oss << paths[i].string();
    }
    return oss.str();
}

std::vector<std::filesystem::path> getBlobSearchPaths(const std::filesystem::path& blobPath) {
    if (blobPath.is_absolute()) {
        return {blobPath};
    }

    std::vector<std::filesystem::path> candidates;
    candidates.push_back(blobPath);

    if (const auto* skipCfg = std::getenv("OV_NPU_TESTS_SKIP_CONFIG_FILE"); skipCfg != nullptr) {
        const std::filesystem::path skipCfgPath(skipCfg);
        if (skipCfgPath.has_parent_path()) {
            candidates.push_back(skipCfgPath.parent_path() / blobPath);
        }
    }

    return candidates;
}

std::filesystem::path resolveExistingPathOrThrow(const std::filesystem::path& inputPath,
                                                 const std::string& resourceDescription) {
    const auto candidatePaths = getBlobSearchPaths(inputPath);

    for (const auto& candidate : candidatePaths) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    throw std::runtime_error("Cannot open " + resourceDescription + ": " + inputPath.string() +
                             ". Checked paths: " + formatCheckedPaths(candidatePaths));
}

}  // namespace

using NPUVMRuntimeCAPITestParams = std::tuple<std::string,  // Device name
                                              std::string,  // Model path or name
                                              std::string,  // Library name
                                              ov::AnyMap    // Config
                                              >;

class NPUVMRuntimeCAPITest :
        public testing::WithParamInterface<NPUVMRuntimeCAPITestParams>,
        public ov::test::TestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NPUVMRuntimeCAPITestParams>& obj) {
        std::string targetDevice;
        std::string modelPath;
        std::string libName;
        ov::AnyMap configuration;

        std::tie(targetDevice, modelPath, libName, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

        std::ostringstream result;
        result << "target_platform=" <<  ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        result << "model=" << modelPath << "_";
        result << "target_device=" << targetDevice << "_";
        result << "library_name=" << libName << "_";

        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

protected:
    void checkStatus(const std::string& step, const ze_result_t result) {
        if (ZE_RESULT_SUCCESS != result) {
            throw std::runtime_error("L0 " + step + " failed: " + intel_npu::ze_result_to_string(result) + ", code 0x" +
                                     std::to_string(result) + " - " + intel_npu::ze_result_to_description(result));
        }
    }

    void SetUp() override {
        // Skip test according to plugin specific disabled_test_patterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        try {
            std::string targetDevice;
            std::string modelPath;
            std::string libName;
            ov::AnyMap configuration;

            std::tie(targetDevice, modelPath, libName, configuration) = this->GetParam();

            auto libpath = ov::util::make_plugin_library_name({}, libName);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            lib = ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#else
            lib = ov::util::load_shared_object(libpath.c_str());
#endif
            fntbl.init(lib, libName);

            ov::Core core;
            const auto deviceNames =
                    core.get_property("NPU", ov::available_devices.name()).as<std::vector<std::string>>();
            const bool deviceAvailable =
                    std::any_of(deviceNames.begin(), deviceNames.end(), [&](const std::string& name) {
                        return targetDevice.find(name) != std::string::npos;
                    });
            if (!deviceAvailable) {
                std::cerr << "Warning: Target device " << targetDevice
                          << " not found in available devices: " << ov::util::join(deviceNames) << std::endl;
                GTEST_SKIP() << "Skip test for current device";
            }

            const auto ext = std::filesystem::path(modelPath).extension().string();
            if (ext == ".xml") {
                compileBlobFromOvIR(core, modelPath, targetDevice, configuration);
            } else if (ext == ".blob") {
                loadBlobFromFile(modelPath);
            } else {
                compileBlobFromCustomModel(core, modelPath, targetDevice, configuration);
            }

            if (blob.str().empty()) {
                std::cerr << "[SetUp] Cannot export blob" << std::endl;
                GTEST_FAIL() << "Cannot export blob";
            }
        } catch (const std::runtime_error& error) {
            std::cerr << "[SetUp] std::runtime_error: " << error.what() << std::endl;
            GTEST_FAIL() << error.what();
        }
    }

    void compileBlobFromOvIR(ov::Core& core, const std::string& modelPath, const std::string& targetDevice,
                             const ov::AnyMap& config) {
        const auto resolvedPath = resolveExistingPathOrThrow(std::filesystem::path(modelPath), "IR model");
        auto model = core.read_model(resolvedPath.string());
        auto preprocessor = ov::preprocess::PrePostProcessor(model);
        for (size_t i = 0; i < model->inputs().size(); i++) {
            preprocessor.input(i).tensor().set_element_type(ov::element::f16);
        }
        for (size_t i = 0; i < model->outputs().size(); i++) {
            preprocessor.output(i).tensor().set_element_type(ov::element::f16);
        }
        auto compiledModel = core.compile_model(preprocessor.build(), targetDevice, config);
        compiledModel.export_model(blob);
    }

    // Reads a pre-compiled blob file directly into the blob stream, bypassing compilation.
    void loadBlobFromFile(const std::string& blobPath) {
        const auto resolvedPath = resolveExistingPathOrThrow(std::filesystem::path(blobPath), "blob file");

        std::ifstream file(resolvedPath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to read blob file: " + resolvedPath.string());
        }
        blob << file.rdbuf();
    }

    // Builds a named model via the OpenVINO C++ API, compiles, and exports to blob.
    void compileBlobFromCustomModel(ov::Core& core, const std::string& modelName, const std::string& targetDevice,
                                    const ov::AnyMap& config) {
        static const std::unordered_map<std::string, std::function<std::shared_ptr<ov::Model>()>> kBuilders = {
                {"MaxPool", createMaxPoolModel},
        };
        auto it = kBuilders.find(modelName);
        if (it == kBuilders.end()) {
            throw std::runtime_error("Unknown custom model name: " + modelName);
        }
        auto compiledModel = core.compile_model(it->second(), targetDevice, config);
        compiledModel.export_model(blob);
    }

    using ZeroMemRef = std::pair<std::shared_ptr<intel_npu::ZeroMem>, npu_vm_runtime_mem_ref_handle_t>;

    // Creates a runtime handle from the compiled blob. Uses ASSERT to abort on failure so callers
    // receive a valid, non-null handle.
    void createRuntimeHandle(npu_vm_runtime_handle_t& handle, npu_vm_runtime_properties_t& props) {
        const std::string content = blob.str();
        npu_vm_runtime_blob_desc_t blobDesc;
        blobDesc.pInput = reinterpret_cast<const uint8_t*>(content.data());
        blobDesc.inputSize = content.size();
        ASSERT_EQ(fntbl.npuVMRuntimeCreate(&blobDesc, &handle, &props), NPU_VM_RUNTIME_RESULT_SUCCESS);
        ASSERT_NE(handle, nullptr);
    }

    // Allocates ZeroMem-backed memref handles for all graph arguments.
    void prepareZeroMemRefs(npu_vm_runtime_handle_t handle, const npu_vm_runtime_properties_t& props,
                            const std::shared_ptr<intel_npu::ZeroInitStructsHolder>& initStruct,
                            std::vector<ZeroMemRef>& inputs, std::vector<ZeroMemRef>& outputs) {
        constexpr std::size_t kPageSize = 4096;
        ze_graph_argument_properties_3_t arg;
        ze_graph_argument_metadata_t meta;
        int64_t upperBound[5];
        for (uint32_t i = 0; i < props.numOfGraphArgs; i++) {
            ASSERT_EQ(fntbl.npuVMRuntimeGetMetadata(handle, i, &arg, &meta, upperBound), NPU_VM_RUNTIME_RESULT_SUCCESS);

            ov::Shape shapeFromCompiler;
            for (uint32_t d = 0; d < arg.dims_count; d++) {
                shapeFromCompiler.push_back(arg.dims[d]);
            }

            ov::element::Type_t precision = intel_npu::zeroUtils::toOVElementType(arg.devicePrecision);
            ov::element::Type elementType(precision);
            size_t totalSize = std::accumulate(shapeFromCompiler.begin(), shapeFromCompiler.end(), elementType.size(),
                                               [](size_t a, size_t b) {
                                                   return a * b;
                                               });

            std::shared_ptr<intel_npu::ZeroMem> memPtr = intel_npu::zero_mem::allocate_memory(
                    initStruct, totalSize, kPageSize, arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT);

            void* pData = memPtr->data();
            std::vector<int64_t> tensorSize(shapeFromCompiler.begin(), shapeFromCompiler.end());
            std::vector<int64_t> tensorStride(tensorSize.size());
            int64_t stride = 1;
            for (int64_t j = static_cast<int64_t>(tensorSize.size()) - 1; j >= 0; j--) {
                tensorStride[j] = stride;
                stride *= tensorSize[j];
            }

            npu_vm_runtime_mem_ref_handle_t hMemRef;
            ASSERT_EQ(fntbl.npuVMRuntimeCreateMemRef(static_cast<int64_t>(tensorSize.size()), &hMemRef),
                      NPU_VM_RUNTIME_RESULT_SUCCESS);
            ASSERT_EQ(fntbl.npuVMRuntimeSetMemRef(hMemRef, pData, pData, 0, tensorSize.data(), tensorStride.data(),
                                                  static_cast<int64_t>(tensorSize.size())),
                      NPU_VM_RUNTIME_RESULT_SUCCESS);

            if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
                inputs.emplace_back(memPtr, hMemRef);
            } else if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
                outputs.emplace_back(memPtr, hMemRef);
            }
        }
    }

    // Extracts raw handles from ZeroMemRef pairs into a flat vector.
    static void extractHandles(const std::vector<ZeroMemRef>& memRefs,
                               std::vector<npu_vm_runtime_mem_ref_handle_t>& handles) {
        handles.reserve(memRefs.size());
        for (const auto& [mem, handle] : memRefs) {
            handles.push_back(handle);
        }
    }

    // Initialises a command queue, graph extension table, command lists, and a fence.
    void setupZEInfrastructure(const std::shared_ptr<intel_npu::ZeroInitStructsHolder>& initStruct,
                               const npu_vm_runtime_properties_t& props,
                               std::shared_ptr<intel_npu::CommandQueue>& commandQueue,
                               ze_graph_dditable_ext_t*& graphDdiTableExt,
                               ze_command_queue_npu_dditable_ext_t*& commandQueueDdiTableExt,
                               std::vector<std::shared_ptr<intel_npu::CommandList>>& commandLists,
                               std::vector<ze_command_list_handle_t>& commandListHandles,
                               std::shared_ptr<intel_npu::Fence>& fence) {
        const intel_npu::CommandQueueDesc commandQueueDesc{
                ZE_COMMAND_QUEUE_PRIORITY_NORMAL,  // priority
                std::nullopt,                      // workload
                0,                                 // options
                nullptr,                           // owner_tag
                true,                              // shared_common_queue
        };

        commandQueue = std::make_shared<intel_npu::CommandQueue>(initStruct, commandQueueDesc);

        graphDdiTableExt = nullptr;
        checkStatus("zeDriverGetExtensionFunctionAddress",
                    intel_npu::zeDriverGetExtensionFunctionAddress(initStruct->getDriver(), ZE_GRAPH_EXT_NAME,
                                                                   reinterpret_cast<void**>(&graphDdiTableExt)));

        commandQueueDdiTableExt = nullptr;
        checkStatus(
                "zeDriverGetExtensionFunctionAddress",
                intel_npu::zeDriverGetExtensionFunctionAddress(initStruct->getDriver(), ZE_COMMAND_QUEUE_NPU_EXT_NAME,
                                                               reinterpret_cast<void**>(&commandQueueDdiTableExt)));

        commandLists.reserve(props.numOfSubGraphs);
        for (size_t i = 0; i < props.numOfSubGraphs; ++i) {
            commandLists.emplace_back(std::make_shared<intel_npu::CommandList>(initStruct));
        }

        commandListHandles.reserve(commandLists.size());
        for (const auto& cl : commandLists) {
            commandListHandles.push_back(cl->handle());
        }

        fence = std::make_shared<intel_npu::Fence>(commandQueue);
    }

    // Builds execution parameters from the provided ZE handles and memref arg arrays.
    static npu_vm_runtime_execute_params_t buildExecParams(
            const std::shared_ptr<intel_npu::ZeroInitStructsHolder>& initStruct,
            std::vector<npu_vm_runtime_mem_ref_handle_t>& inputArgs,
            std::vector<npu_vm_runtime_mem_ref_handle_t>& outputArgs,
            std::vector<ze_command_list_handle_t>& commandListHandles,
            const std::shared_ptr<intel_npu::CommandQueue>& commandQueue,
            const std::shared_ptr<intel_npu::Fence>& fence, ze_graph_dditable_ext_t* graphDdiTableExt,
            npu_vm_runtime_execution_context_handle_t executionContext = nullptr) {
        npu_vm_runtime_execute_params_t execParams;
        execParams.numOfInputs = static_cast<uint32_t>(inputArgs.size());
        execParams.pInputs = inputArgs.data();
        execParams.numOfOutputs = static_cast<uint32_t>(outputArgs.size());
        execParams.pOutputs = outputArgs.data();
        execParams.ctx = initStruct->getContext();
        execParams.device = initStruct->getDevice();
        execParams.graphDdiTableExt = graphDdiTableExt;
        execParams.commandLists = commandListHandles.data();
        execParams.numCommandLists = static_cast<uint64_t>(commandListHandles.size());
        execParams.commandQueue = commandQueue->handle();
        execParams.inferenceFence = fence->handle();
        execParams.event = nullptr;
        execParams.executionContext = executionContext;
        return execParams;
    }

    void destroyMemRefs(const std::vector<npu_vm_runtime_mem_ref_handle_t>& handles) {
        for (const auto h : handles) {
            EXPECT_EQ(fntbl.npuVMRuntimeDestroyMemRef(h), NPU_VM_RUNTIME_RESULT_SUCCESS);
        }
    }

    // Creates a runtime handle from the compiled blob. Uses ASSERT to abort on failure so callers
    // receive a valid, non-null handle.
    void createRuntimeHandle_v2(npu_vm_runtime_handle_t& handle, npu_vm_runtime_properties_t& props) {
        const std::string content = blob.str();
        npu_vm_runtime_blob_desc_t blobDesc;
        blobDesc.pInput = reinterpret_cast<const uint8_t*>(content.data());
        blobDesc.inputSize = content.size();
        ASSERT_EQ(fntbl.npuVMRuntimeCreate2(&blobDesc, nullptr, &handle, &props), NPU_VM_RUNTIME_RESULT_SUCCESS);
        ASSERT_NE(handle, nullptr);
    }

    // Builds execution parameters from the provided ZE handles and memref arg arrays.
    static npu_vm_runtime_execute_params2_t buildExecParams_v2(
            std::vector<npu_vm_runtime_mem_ref_handle_t>& inputArgs,
            std::vector<npu_vm_runtime_mem_ref_handle_t>& outputArgs,
            const std::shared_ptr<intel_npu::CommandQueue>& commandQueue, npu_vm_runtime_config_desc_t* pConfig,
            npu_vm_runtime_execution_context_handle_t executionContext) {
        npu_vm_runtime_execute_params2_t execParams;
        execParams.numOfInputs = static_cast<uint32_t>(inputArgs.size());
        execParams.pInputs = inputArgs.data();
        execParams.numOfOutputs = static_cast<uint32_t>(outputArgs.size());
        execParams.pOutputs = outputArgs.data();
        execParams.commandQueue = commandQueue->handle();
        execParams.pConfig = pConfig;
        execParams.executionContext = executionContext;
        return execParams;
    }

    std::stringstream blob;
    std::shared_ptr<void> lib;
    npu_vm_runtime_fntbl fntbl;
};

TEST_P(NPUVMRuntimeCAPITest, GetAPIVersion_Success) {
    npu_vm_runtime_version_t version;
    auto result = fntbl.npuVMRuntimeGetAPIVersion(&version);
    EXPECT_EQ(result, NPU_VM_RUNTIME_RESULT_SUCCESS);

    std::string libName;
    std::tie(std::ignore, std::ignore, libName, std::ignore) = this->GetParam();
    EXPECT_EQ(version, NPU_VM_RUNTIME_VERSION_CURRENT);
}

TEST_P(NPUVMRuntimeCAPITest, GetMetadata_ReturnsValidArgProperties) {
    npu_vm_runtime_handle_t handle = nullptr;
    npu_vm_runtime_properties_t props;
    ASSERT_NO_FATAL_FAILURE(createRuntimeHandle(handle, props));

    ASSERT_GT(props.numOfGraphArgs, 0u);

    for (uint32_t i = 0; i < props.numOfGraphArgs; i++) {
        ze_graph_argument_properties_3_t argProps{};
        ze_graph_argument_metadata_t argMeta{};
        int64_t upperBound[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE]{};

        ASSERT_EQ(fntbl.npuVMRuntimeGetMetadata(handle, i, &argProps, &argMeta, upperBound),
                  NPU_VM_RUNTIME_RESULT_SUCCESS);

        // Type must be either input or output
        EXPECT_TRUE(argProps.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT || argProps.type == ZE_GRAPH_ARGUMENT_TYPE_OUTPUT);

        // Precision must be resolved — UNKNOWN indicates unhandled element type
        EXPECT_NE(argProps.networkPrecision, ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN);
        EXPECT_NE(argProps.devicePrecision, ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN);

        // Name must be non-empty
        EXPECT_GT(strnlen(argProps.name, ZE_MAX_GRAPH_ARGUMENT_NAME), 0u);

        // Rank must be positive
        EXPECT_GT(argProps.dims_count, 0u);

        // Metadata argument type must match properties
        EXPECT_EQ(argMeta.type, argProps.type);
    }

    // Out-of-range index must return an error
    {
        ze_graph_argument_properties_3_t argProps{};
        ze_graph_argument_metadata_t argMeta{};
        int64_t upperBound[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE]{};
        EXPECT_NE(fntbl.npuVMRuntimeGetMetadata(handle, props.numOfGraphArgs, &argProps, &argMeta, upperBound),
                  NPU_VM_RUNTIME_RESULT_SUCCESS);
    }

    // Null handle must return an error
    {
        ze_graph_argument_properties_3_t argProps{};
        ze_graph_argument_metadata_t argMeta{};
        int64_t upperBound[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE]{};
        EXPECT_NE(fntbl.npuVMRuntimeGetMetadata(nullptr, 0, &argProps, &argMeta, upperBound),
                  NPU_VM_RUNTIME_RESULT_SUCCESS);
    }

    EXPECT_EQ(fntbl.npuVMRuntimeDestroy(handle), NPU_VM_RUNTIME_RESULT_SUCCESS);
}

// Verifies that the metadata extracted from bytecode.blob matches the exact values defined in
// tests/lit/NPU/dialect/bytecode/serialization.mlir:
//   - 1 input ("Another example") and 1 output ("arithmetic")
//   - Shape [42, 100, 50], precision FP32
//
// The blob was generated with:
//   ./vpux-translate --split-input-file --platform=NPU5010 --export-bytecode
//       <compiler_repo>/tests/lit/NPU/dialect/bytecode/serialization.mlir -o bytecode.blob
TEST_P(NPUVMRuntimeCAPITest, GetMetadata_ExactValuesForBytecodeBlob) {
    std::string modelPath;
    std::tie(std::ignore, modelPath, std::ignore, std::ignore) = this->GetParam();
    if (modelPath != "bytecode.blob") {
        GTEST_SKIP() << "Test applies only to bytecode.blob";
    }

    npu_vm_runtime_handle_t handle = nullptr;
    npu_vm_runtime_properties_t props;
    ASSERT_NO_FATAL_FAILURE(createRuntimeHandle(handle, props));

    // serialization.mlir metadata section defines 1 input + 1 output = 2 graph args
    ASSERT_EQ(props.numOfGraphArgs, 2u);

    struct ExpectedArg {
        const char* name;
        ze_graph_argument_type_t type;
        uint32_t dimsCount;
        uint32_t dims[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];
    };

    const ExpectedArg expected[] = {
            {"Another example", ZE_GRAPH_ARGUMENT_TYPE_INPUT, 3, {42, 100, 50}},
            {"arithmetic", ZE_GRAPH_ARGUMENT_TYPE_OUTPUT, 3, {42, 100, 50}},
    };

    for (uint32_t i = 0; i < props.numOfGraphArgs; i++) {
        ze_graph_argument_properties_3_t argProps{};
        ze_graph_argument_metadata_t argMeta{};
        int64_t upperBound[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE]{};

        ASSERT_EQ(fntbl.npuVMRuntimeGetMetadata(handle, i, &argProps, &argMeta, upperBound),
                  NPU_VM_RUNTIME_RESULT_SUCCESS);

        EXPECT_STREQ(argProps.name, expected[i].name) << "Arg " << i << " name mismatch";
        EXPECT_EQ(argProps.type, expected[i].type) << "Arg " << i << " type mismatch";
        EXPECT_EQ(argProps.networkPrecision, ZE_GRAPH_ARGUMENT_PRECISION_FP32)
                << "Arg " << i << " networkPrecision mismatch";
        EXPECT_EQ(argProps.devicePrecision, ZE_GRAPH_ARGUMENT_PRECISION_FP32)
                << "Arg " << i << " devicePrecision mismatch";
        EXPECT_EQ(argProps.dims_count, expected[i].dimsCount) << "Arg " << i << " dims_count mismatch";
        for (uint32_t d = 0; d < expected[i].dimsCount; d++) {
            EXPECT_EQ(argProps.dims[d], expected[i].dims[d]) << "Arg " << i << " dim[" << d << "] mismatch";
        }

        EXPECT_EQ(argMeta.type, expected[i].type) << "Arg " << i << " metadata type mismatch";
        EXPECT_STREQ(argMeta.friendly_name, expected[i].name) << "Arg " << i << " friendly_name mismatch";
    }

    EXPECT_EQ(fntbl.npuVMRuntimeDestroy(handle), NPU_VM_RUNTIME_RESULT_SUCCESS);
}

TEST_P(NPUVMRuntimeCAPITest, ExecuteCompiledModel) {
    std::string libName;
    std::string modelPath;
    std::tie(std::ignore, modelPath, libName, std::ignore) = this->GetParam();

    if (modelPath == "bytecode.blob") {
        GTEST_SKIP() << "Test applies only to bytecode.blob";
    }

    npu_vm_runtime_handle_t handle = nullptr;
    npu_vm_runtime_properties_t props;
    ASSERT_NO_FATAL_FAILURE(createRuntimeHandle(handle, props));

    auto initStruct = std::make_shared<intel_npu::ZeroInitStructsHolder>();
    std::vector<ZeroMemRef> inputs, outputs;
    ASSERT_NO_FATAL_FAILURE(prepareZeroMemRefs(handle, props, initStruct, inputs, outputs));

    std::vector<npu_vm_runtime_mem_ref_handle_t> inputArgs, outputArgs;
    extractHandles(inputs, inputArgs);
    extractHandles(outputs, outputArgs);

    std::shared_ptr<intel_npu::CommandQueue> commandQueue;
    ze_graph_dditable_ext_t* graphDdiTableExt = nullptr;
    ze_command_queue_npu_dditable_ext_t* commandQueueDdiTableExt = nullptr;
    std::vector<std::shared_ptr<intel_npu::CommandList>> commandLists;
    std::vector<ze_command_list_handle_t> commandListHandles;
    std::shared_ptr<intel_npu::Fence> fence;
    setupZEInfrastructure(initStruct, props, commandQueue, graphDdiTableExt, commandQueueDdiTableExt, commandLists,
                          commandListHandles, fence);

    npu_vm_runtime_execution_context_handle_t executionContextHandle = nullptr;
    ASSERT_EQ(fntbl.npuVMRuntimeCreateExecutionContext(handle, &executionContextHandle), NPU_VM_RUNTIME_RESULT_SUCCESS);
    ASSERT_NE(executionContextHandle, nullptr);

    auto execParams = buildExecParams(initStruct, inputArgs, outputArgs, commandListHandles, commandQueue, fence,
                                      graphDdiTableExt, executionContextHandle);
    EXPECT_EQ(fntbl.npuVMRuntimeExecute(handle, &execParams), NPU_VM_RUNTIME_RESULT_SUCCESS);
    fence->hostSynchronize();

    // TODO: dump output to check accuracy once more models are supported
    EXPECT_EQ(fntbl.npuVMRuntimeDestroy(handle), NPU_VM_RUNTIME_RESULT_SUCCESS);
    destroyMemRefs(inputArgs);
    destroyMemRefs(outputArgs);
}

TEST_P(NPUVMRuntimeCAPITest, UpdateMutableCommandList) {
    std::string libName;
    std::tie(std::ignore, std::ignore, libName, std::ignore) = this->GetParam();
    // E#211625: Skip update mutable command list support for interpreter runtime is not implemented
    if (libName == "openvino_intel_npu_vm_runtime") {
        GTEST_SKIP() << "UpdateMutableCommandList is temporarily disabled for interpreter runtime";
    }

    npu_vm_runtime_handle_t handle = nullptr;
    npu_vm_runtime_properties_t props;
    ASSERT_NO_FATAL_FAILURE(createRuntimeHandle(handle, props));

    npu_vm_runtime_execution_context_handle_t executionContextHandle = nullptr;
    ASSERT_EQ(fntbl.npuVMRuntimeCreateExecutionContext(handle, &executionContextHandle), NPU_VM_RUNTIME_RESULT_SUCCESS);
    ASSERT_NE(executionContextHandle, nullptr);

    auto initStruct = std::make_shared<intel_npu::ZeroInitStructsHolder>();
    std::vector<ZeroMemRef> inputs, outputs;
    ASSERT_NO_FATAL_FAILURE(prepareZeroMemRefs(handle, props, initStruct, inputs, outputs));

    std::vector<npu_vm_runtime_mem_ref_handle_t> inputArgs, outputArgs;
    extractHandles(inputs, inputArgs);
    extractHandles(outputs, outputArgs);

    std::shared_ptr<intel_npu::CommandQueue> commandQueue;
    ze_graph_dditable_ext_t* graphDdiTableExt = nullptr;
    ze_command_queue_npu_dditable_ext_t* commandQueueDdiTableExt = nullptr;
    std::vector<std::shared_ptr<intel_npu::CommandList>> commandLists;
    std::vector<ze_command_list_handle_t> commandListHandles;
    std::shared_ptr<intel_npu::Fence> fence;
    setupZEInfrastructure(initStruct, props, commandQueue, graphDdiTableExt, commandQueueDdiTableExt, commandLists,
                          commandListHandles, fence);

    auto execParams = buildExecParams(initStruct, inputArgs, outputArgs, commandListHandles, commandQueue, fence,
                                      graphDdiTableExt, executionContextHandle);
    EXPECT_EQ(fntbl.npuVMRuntimeExecute(handle, &execParams), NPU_VM_RUNTIME_RESULT_SUCCESS);
    fence->hostSynchronize();

    std::vector<uint64_t> argIndexArray(props.numOfGraphArgs);
    std::iota(argIndexArray.begin(), argIndexArray.end(), 0u);
    EXPECT_EQ(
            fntbl.npuVMRuntimeUpdateMutableCommandList(handle, &execParams, argIndexArray.data(), argIndexArray.size()),
            NPU_VM_RUNTIME_RESULT_SUCCESS);
    intel_npu::zeCommandQueueExecuteCommandLists(commandQueue->handle(), static_cast<uint32_t>(commandLists.size()), commandListHandles.data(),
                                                 fence->handle());
    fence->hostSynchronize();

    EXPECT_EQ(fntbl.npuVMRuntimeDestroyExecutionContext(executionContextHandle), NPU_VM_RUNTIME_RESULT_SUCCESS);

    // TODO: dump output to check accuracy once more models are supported
    EXPECT_EQ(fntbl.npuVMRuntimeDestroy(handle), NPU_VM_RUNTIME_RESULT_SUCCESS);
    destroyMemRefs(inputArgs);
    destroyMemRefs(outputArgs);
}

// E#227224: PredictOutputShape is temporarily disabled.
TEST_P(NPUVMRuntimeCAPITest, DISABLED_PredictShape) {
    npu_vm_runtime_handle_t handle = nullptr;
    npu_vm_runtime_properties_t props;
    ASSERT_NO_FATAL_FAILURE(createRuntimeHandle(handle, props));

    struct LocalMemRef {
        const void* basePtr;
        const void* data;
        int64_t offset;
        std::vector<int64_t> sizes;
        std::vector<int64_t> strides;
        int64_t dimsCount;
        std::vector<int64_t> dynamicRanks;
    };

    std::vector<LocalMemRef> inputs;
    std::vector<LocalMemRef> outputs;

    ze_graph_argument_properties_3_t arg;
    ze_graph_argument_metadata_t meta;
    int64_t upperBound[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];
    const uint64_t dynamicRankValue = std::numeric_limits<uint64_t>::max();
    for (uint32_t i = 0; i < props.numOfGraphArgs; i++) {
        ASSERT_EQ(fntbl.npuVMRuntimeGetMetadata(handle, i, &arg, &meta, upperBound), NPU_VM_RUNTIME_RESULT_SUCCESS);

        LocalMemRef localMemRef;
        localMemRef.basePtr = nullptr;
        localMemRef.data = nullptr;
        localMemRef.offset = 0;
        localMemRef.dimsCount = arg.dims_count;
        // Upper bound shape — actual size may be smaller
        for (uint32_t d = 0; d < arg.dims_count; d++) {
            localMemRef.sizes.push_back(arg.dims[d]);
        }
        // Dimensions where shape[d] == max uint64 are dynamic
        for (uint32_t d = 0; d < meta.shape_size; d++) {
            if (meta.shape[d] == dynamicRankValue) {
                localMemRef.dynamicRanks.push_back(d);
            }
        }
        // Compute row-major (NCHW) strides
        uint64_t stride = 1;
        localMemRef.strides.resize(localMemRef.dimsCount);
        for (auto d = static_cast<int64_t>(localMemRef.dimsCount - 1); d >= 0; d--) {
            localMemRef.strides[d] = stride;
            stride *= localMemRef.sizes[d];
        }
        if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            inputs.push_back(localMemRef);
        } else if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
            outputs.push_back(localMemRef);
        }
    }

    std::vector<npu_vm_runtime_mem_ref_handle_t> inputArgs;
    std::vector<npu_vm_runtime_mem_ref_handle_t> outputArgs;
    for (auto& input : inputs) {
        npu_vm_runtime_mem_ref_handle_t hMemRef;
        ASSERT_EQ(fntbl.npuVMRuntimeCreateMemRef(input.dimsCount, &hMemRef), NPU_VM_RUNTIME_RESULT_SUCCESS);
        ASSERT_EQ(fntbl.npuVMRuntimeSetMemRef(hMemRef, input.basePtr, input.data, input.offset, input.sizes.data(),
                                              input.strides.data(), input.dimsCount),
                  NPU_VM_RUNTIME_RESULT_SUCCESS);
        inputArgs.push_back(hMemRef);
    }
    for (auto& output : outputs) {
        npu_vm_runtime_mem_ref_handle_t hMemRef;
        ASSERT_EQ(fntbl.npuVMRuntimeCreateMemRef(output.dimsCount, &hMemRef), NPU_VM_RUNTIME_RESULT_SUCCESS);
        ASSERT_EQ(fntbl.npuVMRuntimeSetMemRef(hMemRef, output.basePtr, output.data, output.offset, output.sizes.data(),
                                              output.strides.data(), output.dimsCount),
                  NPU_VM_RUNTIME_RESULT_SUCCESS);
        outputArgs.push_back(hMemRef);
    }
    npu_vm_runtime_predict_output_shape_params_t predictParam;
    predictParam.pInputs = inputArgs.data();
    predictParam.numOfInputs = static_cast<uint32_t>(inputArgs.size());
    predictParam.pOutputs = outputArgs.data();
    predictParam.numOfOutputs = static_cast<uint32_t>(outputArgs.size());
    EXPECT_EQ(fntbl.npuVMRuntimePredictOutputShape(handle, &predictParam), NPU_VM_RUNTIME_RESULT_SUCCESS);
    // Verify the output shape
    for (size_t i = 0; i < outputs.size(); i++) {
        LocalMemRef parsedMemRef;
        parsedMemRef.sizes.resize(outputs[i].dimsCount);
        parsedMemRef.strides.resize(outputs[i].dimsCount);
        EXPECT_EQ(fntbl.npuVMRuntimeParseMemRef(outputArgs[i], &parsedMemRef.basePtr, &parsedMemRef.data,
                                                &parsedMemRef.offset, parsedMemRef.sizes.data(),
                                                parsedMemRef.strides.data(), &parsedMemRef.dimsCount),
                  NPU_VM_RUNTIME_RESULT_SUCCESS);
        EXPECT_EQ(outputs[i].dimsCount, parsedMemRef.dimsCount);

        for (int64_t d = 0; d < outputs[i].dimsCount; d++) {
            EXPECT_TRUE(outputs[i].sizes[d] >= parsedMemRef.sizes[d]);
            EXPECT_TRUE(outputs[i].strides[d] >= parsedMemRef.strides[d]);
        }
    }
    EXPECT_EQ(fntbl.npuVMRuntimeDestroy(handle), NPU_VM_RUNTIME_RESULT_SUCCESS);
    destroyMemRefs(inputArgs);
    destroyMemRefs(outputArgs);
}

// E#227224: PredictOutputShape is temporarily disabled.
TEST_P(NPUVMRuntimeCAPITest, DISABLED_PredictShape2) {
    npu_vm_runtime_handle_t handle = nullptr;
    npu_vm_runtime_properties_t props;
    ASSERT_NO_FATAL_FAILURE(createRuntimeHandle(handle, props));

    npu_vm_runtime_execution_context_handle_t executionContextHandle = nullptr;
    ASSERT_EQ(fntbl.npuVMRuntimeCreateExecutionContext(handle, &executionContextHandle), NPU_VM_RUNTIME_RESULT_SUCCESS);
    ASSERT_NE(executionContextHandle, nullptr);

    struct LocalMemRef {
        const void* basePtr;
        const void* data;
        int64_t offset;
        std::vector<int64_t> sizes;
        std::vector<int64_t> strides;
        int64_t dimsCount;
        std::vector<int64_t> dynamicRanks;
    };

    std::vector<LocalMemRef> inputs;
    std::vector<LocalMemRef> outputs;

    ze_graph_argument_properties_3_t arg;
    ze_graph_argument_metadata_t meta;
    int64_t upperBound[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];
    const uint64_t dynamicRankValue = std::numeric_limits<uint64_t>::max();
    for (uint32_t i = 0; i < props.numOfGraphArgs; i++) {
        ASSERT_EQ(fntbl.npuVMRuntimeGetMetadata(handle, i, &arg, &meta, upperBound), NPU_VM_RUNTIME_RESULT_SUCCESS);

        LocalMemRef localMemRef;
        localMemRef.basePtr = nullptr;
        localMemRef.data = nullptr;
        localMemRef.offset = 0;
        localMemRef.dimsCount = arg.dims_count;
        // Upper bound shape — actual size may be smaller
        for (uint32_t d = 0; d < arg.dims_count; d++) {
            localMemRef.sizes.push_back(arg.dims[d]);
        }
        // Dimensions where shape[d] == max uint64 are dynamic
        for (uint32_t d = 0; d < meta.shape_size; d++) {
            if (meta.shape[d] == dynamicRankValue) {
                localMemRef.dynamicRanks.push_back(d);
            }
        }
        // Compute row-major (NCHW) strides
        uint64_t stride = 1;
        localMemRef.strides.resize(localMemRef.dimsCount);
        for (int64_t d = localMemRef.dimsCount - 1; d >= 0; d--) {
            localMemRef.strides[d] = stride;
            stride *= localMemRef.sizes[d];
        }
        if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            inputs.push_back(localMemRef);
        } else if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
            outputs.push_back(localMemRef);
        }
    }

    std::vector<npu_vm_runtime_mem_ref_handle_t> inputArgs;
    std::vector<npu_vm_runtime_mem_ref_handle_t> outputArgs;
    for (auto& input : inputs) {
        npu_vm_runtime_mem_ref_handle_t hMemRef;
        ASSERT_EQ(fntbl.npuVMRuntimeCreateMemRef(input.dimsCount, &hMemRef), NPU_VM_RUNTIME_RESULT_SUCCESS);
        ASSERT_EQ(fntbl.npuVMRuntimeSetMemRef(hMemRef, input.basePtr, input.data, input.offset, input.sizes.data(),
                                              input.strides.data(), input.dimsCount),
                  NPU_VM_RUNTIME_RESULT_SUCCESS);
        inputArgs.push_back(hMemRef);
    }
    for (auto& output : outputs) {
        npu_vm_runtime_mem_ref_handle_t hMemRef;
        ASSERT_EQ(fntbl.npuVMRuntimeCreateMemRef(output.dimsCount, &hMemRef), NPU_VM_RUNTIME_RESULT_SUCCESS);
        ASSERT_EQ(fntbl.npuVMRuntimeSetMemRef(hMemRef, output.basePtr, output.data, output.offset, output.sizes.data(),
                                              output.strides.data(), output.dimsCount),
                  NPU_VM_RUNTIME_RESULT_SUCCESS);
        outputArgs.push_back(hMemRef);
    }
    npu_vm_runtime_predict_output_shape_params_t2 predictParam;
    predictParam.pInputs = inputArgs.data();
    predictParam.numOfInputs = static_cast<uint32_t>(inputArgs.size());
    predictParam.pOutputs = outputArgs.data();
    predictParam.numOfOutputs = static_cast<uint32_t>(outputArgs.size());
    predictParam.executionContext = executionContextHandle;
    EXPECT_EQ(fntbl.npuVMRuntimePredictOutputShape2(handle, &predictParam), NPU_VM_RUNTIME_RESULT_SUCCESS);
    // Verify the output shape
    for (size_t i = 0; i < outputs.size(); i++) {
        LocalMemRef parsedMemRef;
        parsedMemRef.sizes.resize(outputs[i].dimsCount);
        parsedMemRef.strides.resize(outputs[i].dimsCount);
        EXPECT_EQ(fntbl.npuVMRuntimeParseMemRef(outputArgs[i], &parsedMemRef.basePtr, &parsedMemRef.data,
                                                &parsedMemRef.offset, parsedMemRef.sizes.data(),
                                                parsedMemRef.strides.data(), &parsedMemRef.dimsCount),
                  NPU_VM_RUNTIME_RESULT_SUCCESS);
        EXPECT_EQ(outputs[i].dimsCount, parsedMemRef.dimsCount);

        for (int64_t d = 0; d < outputs[i].dimsCount; d++) {
            EXPECT_TRUE(outputs[i].sizes[d] >= parsedMemRef.sizes[d]);
            EXPECT_TRUE(outputs[i].strides[d] >= parsedMemRef.strides[d]);
        }
    }
    EXPECT_EQ(fntbl.npuVMRuntimeDestroyExecutionContext(executionContextHandle), NPU_VM_RUNTIME_RESULT_SUCCESS);
    EXPECT_EQ(fntbl.npuVMRuntimeDestroy(handle), NPU_VM_RUNTIME_RESULT_SUCCESS);
    destroyMemRefs(inputArgs);
    destroyMemRefs(outputArgs);
}

TEST_P(NPUVMRuntimeCAPITest, GetMetadata_ExactValuesForBytecodeBlob_v2) {
    std::string modelPath;
    std::string libName;
    std::tie(std::ignore, modelPath, libName, std::ignore) = this->GetParam();
    if (modelPath != "bytecode.blob") {
        GTEST_SKIP() << "Test applies only to bytecode.blob";
    }

    npu_vm_runtime_version_t version;
    auto result = fntbl.npuVMRuntimeGetAPIVersion(&version);
    EXPECT_EQ(result, NPU_VM_RUNTIME_RESULT_SUCCESS);

    if (NPU_VM_RUNTIME_MAJOR_VERSION(version) < 2) {
        GTEST_SKIP() << "GetMetadata_ExactValuesForBytecodeBlob_v2 requires NPU VM Runtime v2.0 or later";
    }

    npu_vm_runtime_handle_t handle = nullptr;
    npu_vm_runtime_properties_t props;
    ASSERT_NO_FATAL_FAILURE(createRuntimeHandle_v2(handle, props));

    // serialization.mlir metadata section defines 1 input + 1 output = 2 graph args
    ASSERT_EQ(props.numOfGraphArgs, 2u);

    struct ExpectedArg {
        const char* name;
        ze_graph_argument_type_t type;
        uint32_t dimsCount;
        uint32_t dims[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];
    };

    const ExpectedArg expected[] = {
            {"Another example", ZE_GRAPH_ARGUMENT_TYPE_INPUT, 3, {42, 100, 50}},
            {"arithmetic", ZE_GRAPH_ARGUMENT_TYPE_OUTPUT, 3, {42, 100, 50}},
    };

    for (uint32_t i = 0; i < props.numOfGraphArgs; i++) {
        ze_graph_argument_properties_3_t argProps{};
        ze_graph_argument_metadata_t argMeta{};
        int64_t upperBound[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE]{};

        ASSERT_EQ(fntbl.npuVMRuntimeGetMetadata(handle, i, &argProps, &argMeta, upperBound),
                  NPU_VM_RUNTIME_RESULT_SUCCESS);

        EXPECT_STREQ(argProps.name, expected[i].name) << "Arg " << i << " name mismatch";
        EXPECT_EQ(argProps.type, expected[i].type) << "Arg " << i << " type mismatch";
        EXPECT_EQ(argProps.networkPrecision, ZE_GRAPH_ARGUMENT_PRECISION_FP32)
                << "Arg " << i << " networkPrecision mismatch";
        EXPECT_EQ(argProps.devicePrecision, ZE_GRAPH_ARGUMENT_PRECISION_FP32)
                << "Arg " << i << " devicePrecision mismatch";
        EXPECT_EQ(argProps.dims_count, expected[i].dimsCount) << "Arg " << i << " dims_count mismatch";
        for (uint32_t d = 0; d < expected[i].dimsCount; d++) {
            EXPECT_EQ(argProps.dims[d], expected[i].dims[d]) << "Arg " << i << " dim[" << d << "] mismatch";
        }

        EXPECT_EQ(argMeta.type, expected[i].type) << "Arg " << i << " metadata type mismatch";
        EXPECT_STREQ(argMeta.friendly_name, expected[i].name) << "Arg " << i << " friendly_name mismatch";
    }

    EXPECT_EQ(fntbl.npuVMRuntimeDestroy(handle), NPU_VM_RUNTIME_RESULT_SUCCESS);
}

TEST_P(NPUVMRuntimeCAPITest, ExecuteCompiledModel_v2) {
    std::string libName;
    std::string modelPath;
    std::tie(std::ignore, modelPath, libName, std::ignore) = this->GetParam();

    if (modelPath == "bytecode.blob") {
        GTEST_SKIP() << "Test applies only to bytecode.blob";
    }
    npu_vm_runtime_version_t version;
    auto result = fntbl.npuVMRuntimeGetAPIVersion(&version);
    EXPECT_EQ(result, NPU_VM_RUNTIME_RESULT_SUCCESS);

    if (NPU_VM_RUNTIME_MAJOR_VERSION(version) < 2) {
        GTEST_SKIP() << "ExecuteCompiledModel_v2 requires NPU VM Runtime v2.0 or later";
    }

    npu_vm_runtime_handle_t handle = nullptr;
    npu_vm_runtime_properties_t props;
    ASSERT_NO_FATAL_FAILURE(createRuntimeHandle_v2(handle, props));

    auto initStruct = std::make_shared<intel_npu::ZeroInitStructsHolder>();
    std::vector<ZeroMemRef> inputs, outputs;
    ASSERT_NO_FATAL_FAILURE(prepareZeroMemRefs(handle, props, initStruct, inputs, outputs));

    std::vector<npu_vm_runtime_mem_ref_handle_t> inputArgs, outputArgs;
    extractHandles(inputs, inputArgs);
    extractHandles(outputs, outputArgs);

    std::shared_ptr<intel_npu::CommandQueue> commandQueue;
    ze_graph_dditable_ext_t* graphDdiTableExt = nullptr;
    ze_command_queue_npu_dditable_ext_t* commandQueueDdiTableExt = nullptr;
    std::vector<std::shared_ptr<intel_npu::CommandList>> commandLists;
    std::vector<ze_command_list_handle_t> commandListHandles;
    std::shared_ptr<intel_npu::Fence> fence;
    setupZEInfrastructure(initStruct, props, commandQueue, graphDdiTableExt, commandQueueDdiTableExt, commandLists,
                          commandListHandles, fence);

    npu_vm_runtime_execution_context_handle_t executionContextHandle = nullptr;
    npu_vm_runtime_create_execution_context_params_t createContextParams;
    createContextParams.commandQueue = nullptr;
    createContextParams.ctx = initStruct->getContext();
    createContextParams.device = initStruct->getDevice();
    createContextParams.graphDdiTableExt = graphDdiTableExt;
    createContextParams.commandQueueDdiTableExt = commandQueueDdiTableExt;
    ASSERT_EQ(fntbl.npuVMRuntimeCreateExecutionContext2(handle, &createContextParams, &executionContextHandle),
              NPU_VM_RUNTIME_RESULT_SUCCESS);
    ASSERT_NE(executionContextHandle, nullptr);

    auto execParams =
            buildExecParams_v2(inputArgs, outputArgs, commandQueue, /*pConfig*/ nullptr, executionContextHandle);
    EXPECT_EQ(fntbl.npuVMRuntimeExecute2(handle, &execParams), NPU_VM_RUNTIME_RESULT_SUCCESS);
    EXPECT_EQ(fntbl.npuVMRuntimeHostSync(handle, &execParams), NPU_VM_RUNTIME_RESULT_SUCCESS);

    // TODO: dump output to check accuracy once more models are supported
    EXPECT_EQ(fntbl.npuVMRuntimeDestroy(handle), NPU_VM_RUNTIME_RESULT_SUCCESS);
    destroyMemRefs(inputArgs);
    destroyMemRefs(outputArgs);
}

TEST_P(NPUVMRuntimeCAPITest, ExecuteCompiledModelWithSharedCmdQueue_v2) {
    std::string libName;
    std::string modelPath;
    std::tie(std::ignore, modelPath, libName, std::ignore) = this->GetParam();

    if (modelPath == "bytecode.blob") {
        GTEST_SKIP() << "Test applies only to bytecode.blob";
    }
    npu_vm_runtime_version_t version;
    auto result = fntbl.npuVMRuntimeGetAPIVersion(&version);
    EXPECT_EQ(result, NPU_VM_RUNTIME_RESULT_SUCCESS);

    if (NPU_VM_RUNTIME_MAJOR_VERSION(version) < 2) {
        GTEST_SKIP() << "ExecuteCompiledModel_v2 requires NPU VM Runtime v2.0 or later";
    }

    npu_vm_runtime_handle_t handle = nullptr;
    npu_vm_runtime_properties_t props;
    ASSERT_NO_FATAL_FAILURE(createRuntimeHandle_v2(handle, props));

    auto initStruct = std::make_shared<intel_npu::ZeroInitStructsHolder>();
    std::vector<ZeroMemRef> inputs, outputs;
    ASSERT_NO_FATAL_FAILURE(prepareZeroMemRefs(handle, props, initStruct, inputs, outputs));

    std::vector<npu_vm_runtime_mem_ref_handle_t> inputArgs, outputArgs;
    extractHandles(inputs, inputArgs);
    extractHandles(outputs, outputArgs);

    std::shared_ptr<intel_npu::CommandQueue> commandQueue;
    ze_graph_dditable_ext_t* graphDdiTableExt = nullptr;
    ze_command_queue_npu_dditable_ext_t* commandQueueDdiTableExt = nullptr;
    std::vector<std::shared_ptr<intel_npu::CommandList>> commandLists;
    std::vector<ze_command_list_handle_t> commandListHandles;
    std::shared_ptr<intel_npu::Fence> fence;
    setupZEInfrastructure(initStruct, props, commandQueue, graphDdiTableExt, commandQueueDdiTableExt, commandLists,
                          commandListHandles, fence);

    npu_vm_runtime_execution_context_handle_t executionContextHandle = nullptr;
    npu_vm_runtime_create_execution_context_params_t createContextParams;
    createContextParams.commandQueue = commandQueue->handle();

    createContextParams.ctx = initStruct->getContext();
    createContextParams.device = initStruct->getDevice();
    createContextParams.graphDdiTableExt = graphDdiTableExt;
    createContextParams.commandQueueDdiTableExt = commandQueueDdiTableExt;
    ASSERT_EQ(fntbl.npuVMRuntimeCreateExecutionContext2(handle, &createContextParams, &executionContextHandle),
              NPU_VM_RUNTIME_RESULT_SUCCESS);
    ASSERT_NE(executionContextHandle, nullptr);

    auto execParams =
            buildExecParams_v2(inputArgs, outputArgs, commandQueue, /*pConfig*/ nullptr, executionContextHandle);
    EXPECT_EQ(fntbl.npuVMRuntimeExecute2(handle, &execParams), NPU_VM_RUNTIME_RESULT_SUCCESS);
    EXPECT_EQ(fntbl.npuVMRuntimeHostSync(handle, &execParams), NPU_VM_RUNTIME_RESULT_SUCCESS);

    // TODO: dump output to check accuracy once more models are supported
    EXPECT_EQ(fntbl.npuVMRuntimeDestroy(handle), NPU_VM_RUNTIME_RESULT_SUCCESS);
    destroyMemRefs(inputArgs);
    destroyMemRefs(outputArgs);
}

TEST_P(NPUVMRuntimeCAPITest, ExecuteCompiledModelWithCmdQueueConfig_v2) {
    std::string libName;
    std::string modelPath;
    std::tie(std::ignore, modelPath, libName, std::ignore) = this->GetParam();

    if (modelPath == "bytecode.blob") {
        GTEST_SKIP() << "Test applies only to bytecode.blob";
    }
    npu_vm_runtime_version_t version;
    auto result = fntbl.npuVMRuntimeGetAPIVersion(&version);
    EXPECT_EQ(result, NPU_VM_RUNTIME_RESULT_SUCCESS);

    if (NPU_VM_RUNTIME_MAJOR_VERSION(version) < 2) {
        GTEST_SKIP() << "ExecuteCompiledModel_v2 requires NPU VM Runtime v2.0 or later";
    }

    npu_vm_runtime_handle_t handle = nullptr;
    npu_vm_runtime_properties_t props;
    ASSERT_NO_FATAL_FAILURE(createRuntimeHandle_v2(handle, props));

    auto initStruct = std::make_shared<intel_npu::ZeroInitStructsHolder>();
    std::vector<ZeroMemRef> inputs, outputs;
    ASSERT_NO_FATAL_FAILURE(prepareZeroMemRefs(handle, props, initStruct, inputs, outputs));

    std::vector<npu_vm_runtime_mem_ref_handle_t> inputArgs, outputArgs;
    extractHandles(inputs, inputArgs);
    extractHandles(outputs, outputArgs);

    std::shared_ptr<intel_npu::CommandQueue> commandQueue;
    ze_graph_dditable_ext_t* graphDdiTableExt = nullptr;
    ze_command_queue_npu_dditable_ext_t* commandQueueDdiTableExt = nullptr;
    std::vector<std::shared_ptr<intel_npu::CommandList>> commandLists;
    std::vector<ze_command_list_handle_t> commandListHandles;
    std::shared_ptr<intel_npu::Fence> fence;
    setupZEInfrastructure(initStruct, props, commandQueue, graphDdiTableExt, commandQueueDdiTableExt, commandLists,
                          commandListHandles, fence);

    npu_vm_runtime_execution_context_handle_t executionContextHandle = nullptr;
    npu_vm_runtime_create_execution_context_params_t createContextParams;
    createContextParams.commandQueue = nullptr;
    createContextParams.ctx = initStruct->getContext();
    createContextParams.device = initStruct->getDevice();
    createContextParams.graphDdiTableExt = graphDdiTableExt;
    createContextParams.commandQueueDdiTableExt = commandQueueDdiTableExt;

    npu_vm_runtime_config_desc_t configDesc[2];
    configDesc[0].type = NPU_VM_RUNTIME_CONFIG_TYPE_QUEUE_PRIORITY;
    configDesc[0].value = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
    configDesc[0].pNext = &configDesc[1];
    configDesc[1].type = NPU_VM_RUNTIME_CONFIG_TYPE_WORKLOAD_TYPE;
    configDesc[1].type = ZE_WORKLOAD_TYPE_BACKGROUND;
    configDesc[1].pNext = nullptr;
    ASSERT_EQ(fntbl.npuVMRuntimeCreateExecutionContext2(handle, &createContextParams, &executionContextHandle),
              NPU_VM_RUNTIME_RESULT_SUCCESS);
    ASSERT_NE(executionContextHandle, nullptr);

    auto execParams = buildExecParams_v2(inputArgs, outputArgs, commandQueue, &configDesc[0], executionContextHandle);
    EXPECT_EQ(fntbl.npuVMRuntimeExecute2(handle, &execParams), NPU_VM_RUNTIME_RESULT_SUCCESS);
    EXPECT_EQ(fntbl.npuVMRuntimeHostSync(handle, &execParams), NPU_VM_RUNTIME_RESULT_SUCCESS);

    npu_vm_runtime_config_desc_t configDesc2{};
    configDesc2.type = NPU_VM_RUNTIME_CONFIG_TYPE_QUEUE_PRIORITY;
    configDesc2.value = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    execParams.pConfig = &configDesc2;
    EXPECT_EQ(fntbl.npuVMRuntimeExecute2(handle, &execParams), NPU_VM_RUNTIME_RESULT_SUCCESS);
    EXPECT_EQ(fntbl.npuVMRuntimeHostSync(handle, &execParams), NPU_VM_RUNTIME_RESULT_SUCCESS);
    // TODO: dump output to check accuracy once more models are supported
    EXPECT_EQ(fntbl.npuVMRuntimeDestroy(handle), NPU_VM_RUNTIME_RESULT_SUCCESS);
    destroyMemRefs(inputArgs);
    destroyMemRefs(outputArgs);
}

const std::vector<std::string> devices = {"NPU.4000", "NPU.5010", "NPU.6010"};

const std::vector<std::string> interpreterModels = {
        // E#214461: Temporarily disabled because IR is unavailable in CI environment.
        // "CustomNet_canonical_strides_1x1_no_fork.xml",
        "bytecode.blob", "MaxPool"};

const std::vector<std::string> interpreterLibNames = {"openvino_intel_npu_vm_runtime"};

const std::vector<ov::AnyMap> configsHostCompile = {
        {{"NPU_COMPILER_TYPE", "PLUGIN"}, {"NPU_COMPILATION_MODE", "HostCompile_Interpreter"}}};
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest_interpreter_runtime, NPUVMRuntimeCAPITest,
                         ::testing::Combine(::testing::ValuesIn(devices), ::testing::ValuesIn(interpreterModels),
                                            ::testing::ValuesIn(interpreterLibNames),
                                            ::testing::ValuesIn(configsHostCompile)),
                         NPUVMRuntimeCAPITest::getTestCaseName);
