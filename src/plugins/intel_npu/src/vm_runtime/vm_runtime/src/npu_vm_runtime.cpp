//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vm_runtime/npu_vm_runtime.hpp"
#include "intel_npu/config/config.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/network_description.hpp"
#include "utils/network_metadata.hpp"
#include "utils/parameters.hpp"
#include "vm_runtime/virtual_machine.h"

#include <ze_command_queue_npu_ext.h>
#include <openvino/core/type/element_type.hpp>

#include <ze_graph_ext.h>
#include "intel_npu/utils/zero/zero_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

/// Opaque runtime object holding the VirtualMachine instance.
struct _npu_vm_runtime_handle_t {
    intel_npu::vm::NetworkMetadata metadata;
    npu_vm_module* module{};
    // Ze command queue handle for executing the graph.
    // A command queue will be created when npuVMRuntimeCreateExecutionContext2 is called with
    // a null command queue
    ze_device_handle_t device;
    ze_context_handle_t context;
    npu_vm_command_queue_desc commandQueueDesc{};
    ze_command_queue_handle_t commandQueue{};
    std::vector<ze_command_queue_handle_t> commandQueues;
    ze_command_queue_npu_dditable_ext_t* commandQueueDdiTableExt{nullptr};
    std::mutex mutex;
};

namespace {
constexpr uint32_t MAX_COMMAND_QUEUE_PRIORITIES = 3;
constexpr uint32_t MAX_GRAPH_ARG_DIMS = ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE;
constexpr uint32_t MAX_GRAPH_TENSOR_REF_DIMS = ZE_MAX_GRAPH_TENSOR_REF_DIMS;

ze_graph_argument_precision_t toZeGraphPrecision(const ov::element::Type& precision) {
    static const std::vector<std::pair<ov::element::Type, ze_graph_argument_precision_t>> precisions = {
            {ov::element::f64, ZE_GRAPH_ARGUMENT_PRECISION_FP64},
            {ov::element::f32, ZE_GRAPH_ARGUMENT_PRECISION_FP32},
            {ov::element::f16, ZE_GRAPH_ARGUMENT_PRECISION_FP16},
            {ov::element::bf16, ZE_GRAPH_ARGUMENT_PRECISION_BF16},
            {ov::element::u64, ZE_GRAPH_ARGUMENT_PRECISION_UINT64},
            {ov::element::u32, ZE_GRAPH_ARGUMENT_PRECISION_UINT32},
            {ov::element::u16, ZE_GRAPH_ARGUMENT_PRECISION_UINT16},
            {ov::element::u8, ZE_GRAPH_ARGUMENT_PRECISION_UINT8},
            {ov::element::u4, ZE_GRAPH_ARGUMENT_PRECISION_UINT4},
            {ov::element::u1, ZE_GRAPH_ARGUMENT_PRECISION_BIN},
            {ov::element::i64, ZE_GRAPH_ARGUMENT_PRECISION_INT64},
            {ov::element::i32, ZE_GRAPH_ARGUMENT_PRECISION_INT32},
            {ov::element::i16, ZE_GRAPH_ARGUMENT_PRECISION_INT16},
            {ov::element::i8, ZE_GRAPH_ARGUMENT_PRECISION_INT8},
            {ov::element::i4, ZE_GRAPH_ARGUMENT_PRECISION_INT4},
            {ov::element::nf4, ZE_GRAPH_ARGUMENT_PRECISION_NF4},
            {ov::element::boolean, ZE_GRAPH_ARGUMENT_PRECISION_BOOLEAN},
    };

    const auto it = std::find_if(precisions.begin(), precisions.end(), [&](const auto& entry) {
        return entry.first == precision;
    });
    return it != precisions.end() ? it->second : ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN;
}

void copyString(char* dst, size_t dstSize, const std::string& src) {
    if (dst == nullptr || dstSize == 0) {
        return;
    }

    std::strncpy(dst, src.c_str(), dstSize - 1);
    dst[dstSize - 1] = '\0';  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

ze_result_t createCmdQueue(npu_vm_runtime_handle_t hRuntime, ze_command_queue_handle_t* commandQueue) {
    auto context = hRuntime->context;
    auto device = hRuntime->device;
    auto& desc = hRuntime->commandQueueDesc;
    auto commandQueueDdiTableExt = hRuntime->commandQueueDdiTableExt;

    if (context == nullptr || device == nullptr) {
        return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (desc.workload_type != ZE_WORKLOAD_TYPE_DEFAULT) {
        if (commandQueueDdiTableExt == nullptr || commandQueueDdiTableExt->pfnSetWorkloadType == nullptr) {
            return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    }

    auto groupOrdinal =
            intel_npu::zeroUtils::findCommandQueueGroupOrdinal(device, ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

    ze_command_queue_desc_t ze_queue_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr,      groupOrdinal, 0, 0,
                                             ZE_COMMAND_QUEUE_MODE_DEFAULT,        desc.priority};
    ze_command_queue_desc_npu_ext_2_t command_queue_desc = {};

    if (desc.option_flags) {
        command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC_NPU_EXT_2;
        command_queue_desc.options = desc.option_flags;
        ze_queue_desc.pNext = &command_queue_desc;
    }

    auto handle = zeCommandQueueCreate(context, device, &ze_queue_desc, commandQueue);

    if (desc.workload_type != ZE_WORKLOAD_TYPE_DEFAULT) {
        auto result = commandQueueDdiTableExt->pfnSetWorkloadType(
                *commandQueue, static_cast<ze_command_queue_workload_type_t>(desc.workload_type));

        if (ZE_RESULT_SUCCESS != result) {
            zeCommandQueueDestroy(*commandQueue);
            *commandQueue = nullptr;
            return result;
        }
    }

    return handle;
}

ze_result_t configCmdQueue(npu_vm_runtime_handle_t hRuntime, const npu_vm_runtime_config_desc_t* pConfig) {
    const npu_vm_runtime_config_desc_t* config = pConfig;
    auto& desc = hRuntime->commandQueueDesc;

    ze_command_queue_priority_t priority = desc.priority;
    auto workloadType = desc.workload_type;
    while (config) {
        if (config->type == NPU_VM_RUNTIME_CONFIG_TYPE_QUEUE_PRIORITY) {
            if (config->value >= MAX_COMMAND_QUEUE_PRIORITIES) {
                NPU_VM_LOG_ERROR("Invalid queue priority value: {}", config->value);
                return ZE_RESULT_ERROR_INVALID_ARGUMENT;
            }
            priority = static_cast<ze_command_queue_priority_t>(config->value);
        } else if (config->type == NPU_VM_RUNTIME_CONFIG_TYPE_WORKLOAD_TYPE) {
            workloadType = static_cast<ze_command_queue_workload_type_t>(config->value);
        } else if (config->type == NPU_VM_RUNTIME_CONFIG_TYPE_QUEUE_OPTIONS) {
            NPU_VM_LOG_ERROR("NPU_VM_RUNTIME_CONFIG_TYPE_QUEUE_OPTIONS is not supported");
            return ZE_RESULT_ERROR_INVALID_ARGUMENT;
        } else {
            NPU_VM_LOG_WARN("Unknown config type: {}", config->type);
        }

        config = config->pNext;
    }

    if (priority != desc.priority || workloadType != desc.workload_type) {
        std::lock_guard<std::mutex> lock(hRuntime->mutex);
        if (priority != desc.priority) {
            desc.priority = priority;
            if (hRuntime->commandQueues[static_cast<uint32_t>(priority)] != nullptr) {
                hRuntime->commandQueue = hRuntime->commandQueues[static_cast<uint32_t>(priority)];
            } else {
                // Create an internal command queue if none is provided in the parameters and none is stored in
                // the runtime handle
                ze_command_queue_handle_t commandQueue = nullptr;
                auto result = createCmdQueue(hRuntime, &commandQueue);
                if (result != ZE_RESULT_SUCCESS) {
                    return result;
                }
                // Stores the command queue according to priority [0..2]
                hRuntime->commandQueues[static_cast<uint32_t>(hRuntime->commandQueueDesc.priority)] = commandQueue;
                hRuntime->commandQueue = commandQueue;
            }
        }
        if (hRuntime->commandQueue == nullptr) {
            NPU_VM_LOG_ERROR("Invalid null command queue pointer");
            return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
        }
        if (workloadType != desc.workload_type) {
            auto commandQueueDdiTableExt = hRuntime->commandQueueDdiTableExt;
            if (commandQueueDdiTableExt == nullptr || commandQueueDdiTableExt->pfnSetWorkloadType == nullptr) {
                NPU_VM_LOG_ERROR("pfnSetWorkloadType is not supported");
                return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
            }

            auto commandQueue = hRuntime->commandQueue;
            desc.workload_type = workloadType;
            auto result = commandQueueDdiTableExt->pfnSetWorkloadType(commandQueue, workloadType);
            if (result != ZE_RESULT_SUCCESS) {
                NPU_VM_LOG_ERROR("Failed to set workload type: {}", result);
                return result;
            }
        }
    }

    return ZE_RESULT_SUCCESS;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__((visibility("default")))
#endif

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeGetAPIVersion(npu_vm_runtime_version_t* pVersion) {
    if (pVersion == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    npu_vm_runtime_version_t version = NPU_VM_RUNTIME_VERSION_CURRENT;

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    // Debug purpose to verify old version of runtime interfaces
    // The env variable can be set to "V1.1" to force the runtime to use the old version.
    const char* envVar = std::getenv("NPU_VM_RUNTIME_VERSION");
    if (envVar != nullptr) {
        std::string modeStr(envVar);
        if (modeStr == "V1.1") {
            version = NPU_VM_RUNTIME_VERSION_1_1;
        }
    }
#endif
    NPU_VM_LOG_INFO("VMRuntime APIVersion {}_{}", NPU_VM_RUNTIME_MAJOR_VERSION(version),
                    NPU_VM_RUNTIME_MINOR_VERSION(version));

    *pVersion = version;

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeCreate(const npu_vm_runtime_blob_desc_t* desc,
                                                                            npu_vm_runtime_handle_t* phRuntime,
                                                                            npu_vm_runtime_properties_t* pProperties) {
    if (desc == nullptr || phRuntime == nullptr || pProperties == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    if (desc->pInput == nullptr || desc->inputSize == 0) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        const auto bytecodePtr = desc->pInput;
        const auto bytecodeSize = desc->inputSize;

        auto handle = std::make_unique<_npu_vm_runtime_handle_t>();
        if (npu_vm_parse_module(bytecodePtr, static_cast<uint32_t>(bytecodeSize), &handle->module) != NPU_VM_SUCCESS) {
            return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        }
        auto metadata = intel_npu::vm::getNetworkMetadata(handle->module);
        if (metadata.has_value()) {
            handle->metadata = std::move(metadata.value());
            if (handle->metadata.numCmdLists == 0 ||
                (handle->metadata.inputs.size() + handle->metadata.outputs.size()) == 0) {
                npu_vm_destroy_module(handle->module);
                handle->module = nullptr;
                NPU_VM_LOG_ERROR(
                        "Metadata section is present but contains invalid data (numCmdLists or inputs/outputs)");
                return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
            }
            pProperties->numOfSubGraphs = static_cast<uint32_t>(handle->metadata.numCmdLists);
            pProperties->numOfGraphArgs =
                    static_cast<uint32_t>(handle->metadata.inputs.size() + handle->metadata.outputs.size());
        } else {
            NPU_VM_LOG_WARN("Bytecode blob does not contain metadata section. Set conservatively "
                            "numOfGraphArgs=2 and numOfSubGraphs=1 to allow for basic functionality.");
            pProperties->numOfSubGraphs =
                    1;  // if metadata is missing, assume a single subgraph with all inputs and outputs
            pProperties->numOfGraphArgs = 2;  // at least one input and one output
        }
        *phRuntime = handle.release();
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Exception while creating runtime: {}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeDestroy(npu_vm_runtime_handle_t hRuntime) {
    if (hRuntime == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    auto result = NPU_VM_RUNTIME_RESULT_SUCCESS;
    if (hRuntime->module != nullptr) {
        if (npu_vm_destroy_module(hRuntime->module) != NPU_VM_SUCCESS) {
            NPU_VM_LOG_ERROR("Failed to destroy module");
            result = NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        } else {
            hRuntime->module = nullptr;
        }
    }

    for (auto& commandQueue : hRuntime->commandQueues) {
        if (commandQueue != nullptr) {
            auto cmdQueueDestroyResult = zeCommandQueueDestroy(commandQueue);
            if (cmdQueueDestroyResult != ZE_RESULT_SUCCESS) {
                NPU_VM_LOG_ERROR("Failed to destroy command queue: {}", cmdQueueDestroyResult);
                result = NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
            } else {
                commandQueue = nullptr;
            }
        }
    }

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    delete hRuntime;
    return result;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeGetMetadata(
        npu_vm_runtime_handle_t hRuntime, uint32_t argIndex, ze_graph_argument_properties_3_t* pGraphArgumentProperties,
        ze_graph_argument_metadata_t* pGraphArgumentMetadata, int64_t* upperBound) {
    if (hRuntime == nullptr || pGraphArgumentProperties == nullptr || pGraphArgumentMetadata == nullptr ||
        upperBound == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        const intel_npu::vm::IODescriptor* descriptor = nullptr;
        ze_graph_argument_type_t argType = ZE_GRAPH_ARGUMENT_TYPE_INPUT;

        const auto inputCount = hRuntime->metadata.inputs.size();
        const auto outputCount = hRuntime->metadata.outputs.size();

        if (argIndex < inputCount) {
            descriptor = &hRuntime->metadata.inputs.at(argIndex);
            argType = ZE_GRAPH_ARGUMENT_TYPE_INPUT;
        } else if (argIndex < inputCount + outputCount) {
            descriptor = &hRuntime->metadata.outputs.at(argIndex - inputCount);
            argType = ZE_GRAPH_ARGUMENT_TYPE_OUTPUT;
        } else {
            return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        }

        *pGraphArgumentProperties = {};
        pGraphArgumentProperties->stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3;
        pGraphArgumentProperties->pNext = nullptr;
        pGraphArgumentProperties->type = argType;
        pGraphArgumentProperties->networkPrecision = toZeGraphPrecision(descriptor->precision);
        pGraphArgumentProperties->devicePrecision = toZeGraphPrecision(descriptor->precision);
        pGraphArgumentProperties->networkLayout = ZE_GRAPH_ARGUMENT_LAYOUT_ANY;
        pGraphArgumentProperties->deviceLayout = ZE_GRAPH_ARGUMENT_LAYOUT_ANY;
        pGraphArgumentProperties->quantReverseScale = 1.0f;
        pGraphArgumentProperties->quantZeroPoint = 0;
        pGraphArgumentProperties->associated_tensor_names_count = 0;

        // The ze_graph_* properties expose fixed-size C arrays and `upperBound` is a raw pointer parameter, so
        // array-to-pointer decay and indexed access are unavoidable when populating them.
        // NOLINTBEGIN
        copyString(pGraphArgumentProperties->name, ZE_MAX_GRAPH_ARGUMENT_NAME, descriptor->nameFromCompiler);
        copyString(pGraphArgumentProperties->debug_friendly_name, ZE_MAX_GRAPH_ARGUMENT_NAME,
                   descriptor->nodeFriendlyName.empty() ? descriptor->nameFromCompiler : descriptor->nodeFriendlyName);

        for (uint32_t i = 0; i < MAX_GRAPH_ARG_DIMS; ++i) {
            pGraphArgumentProperties->dims[i] = 0;
            upperBound[i] = 0;
        }

        const auto rank = descriptor->shapeFromCompiler.rank();
        if (rank.is_static()) {
            const auto rankLength = static_cast<size_t>(rank.get_length());
            const auto cappedRank = std::min(rankLength, static_cast<size_t>(MAX_GRAPH_ARG_DIMS));
            pGraphArgumentProperties->dims_count = static_cast<uint32_t>(cappedRank);

            uint32_t dimIdx = 0;
            for (const auto& dim : descriptor->shapeFromCompiler) {
                if (dimIdx >= pGraphArgumentProperties->dims_count) {
                    break;
                }
                if (!dim.is_dynamic() && dim.get_length() >= 0 &&
                    static_cast<uint64_t>(dim.get_length()) <= std::numeric_limits<uint32_t>::max()) {
                    const auto dimVal = static_cast<uint32_t>(dim.get_length());
                    pGraphArgumentProperties->dims[dimIdx] = dimVal;
                    upperBound[dimIdx] = static_cast<int64_t>(dimVal);
                }
                ++dimIdx;
            }
        } else {
            pGraphArgumentProperties->dims_count = 0;
        }
        // NOLINTEND

        *pGraphArgumentMetadata = {};
        pGraphArgumentMetadata->stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_METADATA;
        pGraphArgumentMetadata->pNext = nullptr;
        pGraphArgumentMetadata->type = argType;
        pGraphArgumentMetadata->data_type = ZE_GRAPH_METADATA_TYPE_UNDEFINED;

        std::vector<std::string> tensorNames(descriptor->outputTensorNames.begin(),
                                             descriptor->outputTensorNames.end());
        std::sort(tensorNames.begin(), tensorNames.end());

        const auto tensorNameCount = static_cast<uint32_t>(
                std::min(tensorNames.size(), static_cast<size_t>(ZE_MAX_GRAPH_TENSOR_NAMES_SIZE)));
        pGraphArgumentMetadata->tensor_names_count = tensorNameCount;
        // The ze_graph_* metadata exposes fixed-size C arrays, so array-to-pointer decay and indexed access are
        // unavoidable when populating them.
        // NOLINTBEGIN
        for (uint32_t i = 0; i < tensorNameCount; ++i) {
            copyString(pGraphArgumentMetadata->tensor_names[i], ZE_MAX_GRAPH_ARGUMENT_NAME, tensorNames.at(i));
        }

        // copy tensors names to associated_tensor_names
        pGraphArgumentProperties->associated_tensor_names_count = pGraphArgumentMetadata->tensor_names_count;
        for (uint32_t i = 0; i < tensorNameCount; i++) {
            copyString(pGraphArgumentProperties->associated_tensor_names[i], ZE_MAX_GRAPH_ARGUMENT_NAME,
                       tensorNames.at(i));
        }

        const auto friendlyName =
                descriptor->nodeFriendlyName.empty() ? descriptor->nameFromCompiler : descriptor->nodeFriendlyName;
        copyString(pGraphArgumentMetadata->friendly_name, ZE_MAX_GRAPH_ARGUMENT_NAME, friendlyName);
        copyString(pGraphArgumentMetadata->input_name, ZE_MAX_GRAPH_ARGUMENT_NAME, descriptor->nameFromCompiler);

        for (uint32_t i = 0; i < MAX_GRAPH_TENSOR_REF_DIMS; ++i) {
            pGraphArgumentMetadata->shape[i] = 0;
        }

        const auto& metadataShape = descriptor->shapeFromIRModel.has_value() ? descriptor->shapeFromIRModel.value()
                                                                             : descriptor->shapeFromCompiler;
        const auto metadataRank = metadataShape.rank();
        if (metadataRank.is_static()) {
            const auto rankLength = static_cast<size_t>(metadataRank.get_length());
            const auto cappedRank = std::min(rankLength, static_cast<size_t>(MAX_GRAPH_TENSOR_REF_DIMS));
            pGraphArgumentMetadata->shape_size = static_cast<uint32_t>(cappedRank);

            uint32_t shapeIdx = 0;
            for (const auto& dim : metadataShape) {
                if (shapeIdx >= pGraphArgumentMetadata->shape_size) {
                    break;
                }
                pGraphArgumentMetadata->shape[shapeIdx] = dim.is_dynamic() ? std::numeric_limits<uint64_t>::max()
                                                                           : static_cast<uint64_t>(dim.get_length());
                ++shapeIdx;
            }
        } else {
            pGraphArgumentMetadata->shape_size = 0;
        }
        // NOLINTEND
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Exception while getting metadata: {}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    } catch (...) {
        NPU_VM_LOG_ERROR("Unknown exception while getting metadata");
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeExecute(npu_vm_runtime_handle_t hRuntime,
                                                                             npu_vm_runtime_execute_params_t* pParams) {
    if (hRuntime == nullptr || pParams == nullptr || pParams->executionContext == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    try {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto engine = reinterpret_cast<npu_vm_engine*>(pParams->executionContext);
        if (npu_vm_reset_state(engine, /*resetExecutionContext=*/true) != NPU_VM_SUCCESS) {
            NPU_VM_LOG_ERROR("Failed to reset engine state before inference");
            return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        }
        if (npu_vm_infer(engine, pParams) != NPU_VM_SUCCESS) {
            NPU_VM_LOG_ERROR("Inference failed for the given parameters");
            return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        }
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Exception while executing model: {}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimePredictOutputShape(
        npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_predict_output_shape_params_t* pParams) {
    if (hRuntime == nullptr || hRuntime->module == nullptr || pParams == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        // Note: Create a temporary instance of the engine to call the output shape prediction function, since this
        // function is called before the execution context is created. Ideally, the execution context should be created
        // before output shape prediction, but it requires changes in NPU plugin flow, so this temporary instance is
        // used. If the execution engine creation is moved before output shape prediction in the future, this code can
        // be simplified by reusing the same engine instance (via npu_vm_runtime_handle_t)
        npu_vm_engine* engine = nullptr;
        if (npu_vm_new_engine(&engine) != NPU_VM_SUCCESS) {
            return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        }

        if (npu_vm_load_module(engine, /*moduleParams*/ nullptr, hRuntime->module) != NPU_VM_SUCCESS) {
            npu_vm_destroy_engine(engine);
            return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        }
        npu_vm_runtime_predict_output_shape_params_t2 params2 = {};
        params2.pInputs = pParams->pInputs;
        params2.numOfInputs = pParams->numOfInputs;
        params2.pOutputs = pParams->pOutputs;
        params2.numOfOutputs = pParams->numOfOutputs;
        // Execution context is not available at this point, and the output shape prediction function should not require
        // it
        params2.executionContext = nullptr;
        if (npu_vm_predict_output_shape(engine, &params2) != NPU_VM_SUCCESS) {
            NPU_VM_LOG_ERROR("Output shape prediction failed for the given parameters");
            npu_vm_destroy_engine(engine);
            return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        }
        if (npu_vm_destroy_engine(engine) != NPU_VM_SUCCESS) {
            NPU_VM_LOG_ERROR("Failed to destroy engine after output shape prediction");
            return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        }
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Exception while executing output shape prediction: {}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimePredictOutputShape2(
        npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_predict_output_shape_params_t2* pParams) {
    if (hRuntime == nullptr || pParams == nullptr || pParams->executionContext == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto* engine = reinterpret_cast<npu_vm_engine*>(pParams->executionContext);
        // Note: The execution context state is not reset, as it will be reset when the main inference takes place (i.e.
        // when npuVMRuntimeExecute is called)
        if (npu_vm_reset_state(engine, /*resetExecutionContext=*/false) != NPU_VM_SUCCESS) {
            NPU_VM_LOG_ERROR("Failed to reset engine state before output shape prediction");
            return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        }
        if (npu_vm_predict_output_shape(engine, pParams) != NPU_VM_SUCCESS) {
            NPU_VM_LOG_ERROR("Output shape prediction failed for the given parameters");
            return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        }
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Exception while executing output shape prediction: {}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeCreateMemRef(int64_t dimsCount, npu_vm_runtime_mem_ref_handle_t* phMemRef) {
    if (phMemRef == nullptr || dimsCount <= 0) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto handle = new (std::nothrow) npu_vm_runtime_mem_ref(dimsCount);
    if (handle == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    *phMemRef = reinterpret_cast<npu_vm_runtime_mem_ref_handle_t>(handle);
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeDestroyMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef) {
    if (hMemRef == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto memRef = reinterpret_cast<npu_vm_runtime_mem_ref*>(hMemRef);
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    delete memRef;
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeSetMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef,
                                                                               const void* basePtr, const void* data,
                                                                               int64_t offset, int64_t* pSizes,
                                                                               int64_t* pStrides, int64_t dimsCount) {
    if (hMemRef == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto memRef = reinterpret_cast<npu_vm_runtime_mem_ref*>(hMemRef);
    if (dimsCount != memRef->dimsCount) {
        return NPU_VM_RUNTIME_RESULT_ERROR_UNSUPPORTED_DIM_COUNT;
    }

    memRef->dirtyFlags = static_cast<uint32_t>(intel_npu::vm::MemRef::DirtyFlag::NONE);
    if (memRef->basePtr != basePtr || memRef->data != data) {
        memRef->basePtr = basePtr;
        memRef->data = data;
        memRef->dirtyFlags |= static_cast<uint32_t>(intel_npu::vm::MemRef::DirtyFlag::BUFFER_PTR);
    }

    if (memRef->offset != offset) {
        memRef->offset = offset;
        memRef->dirtyFlags |= static_cast<uint32_t>(intel_npu::vm::MemRef::DirtyFlag::OFFSET);
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    if (pSizes == nullptr) {
        if (memRef->sizes.size() != static_cast<size_t>(dimsCount)) {
            return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    } else {
        std::copy(pSizes, pSizes + dimsCount, memRef->sizes.begin());
        memRef->dirtyFlags |= static_cast<uint32_t>(intel_npu::vm::MemRef::DirtyFlag::SIZES);
    }

    if (pStrides == nullptr) {
        if (memRef->strides.size() != static_cast<size_t>(dimsCount)) {
            return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
        }
    } else {
        std::copy(pStrides, pStrides + dimsCount, memRef->strides.begin());
        memRef->dirtyFlags |= static_cast<uint32_t>(intel_npu::vm::MemRef::DirtyFlag::STRIDES);
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeParseMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef, const void** pBasePtr, const void** pData,
                        int64_t* pOffset, int64_t* pSizes, int64_t* pStrides, int64_t* pDimsCount) {
    if (hMemRef == nullptr || pBasePtr == nullptr || pData == nullptr || pOffset == nullptr || pSizes == nullptr ||
        pStrides == nullptr || pDimsCount == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto memRef = reinterpret_cast<npu_vm_runtime_mem_ref*>(hMemRef);
    *pBasePtr = memRef->basePtr;
    *pData = memRef->data;
    *pOffset = memRef->offset;
    std::copy(memRef->sizes.begin(), memRef->sizes.end(), pSizes);
    std::copy(memRef->strides.begin(), memRef->strides.end(), pStrides);
    *pDimsCount = memRef->dimsCount;
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeCreateExecutionContext(
        npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_execution_context_handle_t* phExecutionHandle) {
    if (hRuntime == nullptr || hRuntime->module == nullptr || phExecutionHandle == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // The execution context is represented by an instance of the VM engine, whose lifetime is managed by the
    // npuVMRuntimeCreateExecutionContext and npuVMRuntimeDestroyExecutionContext functions. In the end-to-end flow, the
    // engine is created once per inference request, then reused for subsequent inferences
    npu_vm_engine* engine = nullptr;
    if (npu_vm_new_engine(&engine) != NPU_VM_SUCCESS) {
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }
    if (npu_vm_load_module(engine, /*moduleParams*/ nullptr, hRuntime->module) != NPU_VM_SUCCESS) {
        npu_vm_destroy_engine(engine);
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    *phExecutionHandle = reinterpret_cast<npu_vm_runtime_execution_context_handle_t>(engine);
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeDestroyExecutionContext(npu_vm_runtime_execution_context_handle_t phExecutionHandle) {
    if (phExecutionHandle == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto engine = reinterpret_cast<npu_vm_engine*>(phExecutionHandle);
    if (npu_vm_destroy_engine(engine) != NPU_VM_SUCCESS) {
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeUpdateMutableCommandList(npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_execute_params_t* pParams,
                                     uint64_t* argIndexArray, uint64_t /*argIndexArraySize*/) {
    if (hRuntime == nullptr || pParams == nullptr || argIndexArray == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // TODO: implement mutable command list update
    return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeCreate2(const npu_vm_runtime_blob_desc_t* desc, const npu_vm_runtime_config_desc_t* pConfig,
                    npu_vm_runtime_handle_t* phRuntime, npu_vm_runtime_properties_t* pProperties) {
    if (npuVMRuntimeCreate(desc, phRuntime, pProperties) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    const npu_vm_runtime_config_desc_t* curConfig = pConfig;
    auto& cmdQueueDesc = (*phRuntime)->commandQueueDesc;
    (*phRuntime)->commandQueues.resize(MAX_COMMAND_QUEUE_PRIORITIES);
    while (curConfig) {
        switch (curConfig->type) {
        case NPU_VM_RUNTIME_CONFIG_TYPE_QUEUE_PRIORITY: {
            if (curConfig->value >= MAX_COMMAND_QUEUE_PRIORITIES) {
                NPU_VM_LOG_ERROR("Invalid queue priority value: {}", curConfig->value);
                return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_VALUE;
            }
            cmdQueueDesc.priority = static_cast<ze_command_queue_priority_t>(curConfig->value);
            break;
        };
        case NPU_VM_RUNTIME_CONFIG_TYPE_WORKLOAD_TYPE: {
            cmdQueueDesc.workload_type = static_cast<ze_command_queue_workload_type_t>(curConfig->value);
            break;
        };
        case NPU_VM_RUNTIME_CONFIG_TYPE_QUEUE_OPTIONS: {
            cmdQueueDesc.option_flags = static_cast<uint32_t>(curConfig->value);
            break;
        };
        default:
            NPU_VM_LOG_WARN("Unknown config type: {}", curConfig->type);
            break;
        }
        curConfig = curConfig->pNext;
    }
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeCreateExecutionContext2(
        npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_create_execution_context_params_t* pParams,
        npu_vm_runtime_execution_context_handle_t* phExecutionHandle) {
    if (hRuntime == nullptr || hRuntime->module == nullptr || pParams == nullptr || phExecutionHandle == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    hRuntime->device = pParams->device;
    hRuntime->context = pParams->ctx;
    hRuntime->commandQueueDdiTableExt = pParams->commandQueueDdiTableExt;

    if (pParams->commandQueue == nullptr && hRuntime->commandQueue == nullptr) {
        std::lock_guard<std::mutex> lock(hRuntime->mutex);
        // check if command queue was created again
        if (hRuntime->commandQueue == nullptr) {
            // Create an internal command queue if none is provided in the parameters and none is stored in the runtime
            // handle
            ze_command_queue_handle_t commandQueue = nullptr;
            if (createCmdQueue(hRuntime, &commandQueue) != ZE_RESULT_SUCCESS) {
                return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
            }
            // Stores the command queue according to priority [0..2]
            hRuntime->commandQueues[static_cast<uint32_t>(hRuntime->commandQueueDesc.priority)] = commandQueue;
            hRuntime->commandQueue = commandQueue;
        }
    }

    // The execution context is represented by an instance of the VM engine, whose lifetime is managed by the
    // npuVMRuntimeCreateExecutionContext and npuVMRuntimeDestroyExecutionContext functions. In the end-to-end flow, the
    // engine is created once per inference request, then reused for subsequent inferences
    npu_vm_engine* engine = nullptr;
    if (npu_vm_new_engine(&engine) != NPU_VM_SUCCESS) {
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    npu_vm_result result = {};
    npu_vm_module_load_params module_load_params = {};
    module_load_params.concurrentJobCount = 1;
    // The reason for checking numStream != 0, there was a bug to set numStream to 0 in the metadata,
    // which is not correct. The correct value should be 1 for single tile and >1 for multi-tile.
    if (hRuntime->metadata.numStreams != 0) {
        // Checks whether a blob is compiled for single tile.
        ze_device_properties_t deviceProperties{};
        zeDeviceGetProperties(pParams->device, &deviceProperties);
        if (deviceProperties.numSlices > 1 && hRuntime->metadata.numStreams == 1) {
            // The blob is compiled for single tile, but the device has multiple slices.
            // In this case, we need to set the number of tiles to concurrentJobCount
            // for concurrent execution.
            module_load_params.concurrentJobCount = deviceProperties.numSlices;

            NPU_VM_LOG_INFO("Concurrent job count: {}", module_load_params.concurrentJobCount);
        }
    }

    bool counterBasedEventSupported = false;
    // The counter based event is supported only on Windows platform for now.
    // Once all function calls replaced with openvino::zero_loader functions,
    // the counter based event will be supported on Linux platform as well.
#if defined(_WIN32)
    auto graphDdiTableExt = pParams->graphDdiTableExt;
    if (ZE_GRAPH_EXT_VERSION_CURRENT >= ZE_GRAPH_EXT_VERSION_1_19) {
        if (graphDdiTableExt != nullptr && graphDdiTableExt->pfnCompilerIsOptionSupported != nullptr) {
            counterBasedEventSupported = graphDdiTableExt->pfnCompilerIsOptionSupported(
                                                 pParams->device, ZE_NPU_DRIVER_OPTIONS, "COUNTER_BASED_EVENTS",
                                                 nullptr) == ZE_RESULT_SUCCESS;
        }

#if defined(VPUX_DEVELOPER_BUILD)
        const char* envVar = std::getenv("NPU_VM_RUNTIME_DISABLE_COUNTER_BASED_EVENT");
        if (envVar != nullptr) {
            counterBasedEventSupported = !intel_npu::OptionParser<bool>::parse(envVar);
        }
#endif
    }
#endif
    module_load_params.counterBasedEventSupported = counterBasedEventSupported;

    if (pParams->commandQueue == nullptr) {
        // need to use an internally created command queue.
        npu_vm_runtime_create_execution_context_params_t params = *pParams;
        params.commandQueue = hRuntime->commandQueue;
        module_load_params.create_execution_context_params = &params;
        module_load_params.internalCommandQueue = true;
        result = npu_vm_load_module(engine, &module_load_params, hRuntime->module);
    } else {
        // when command queue is shared, npu plugin will pass a valid command queue in pParams,
        // so we can use it directly.
        module_load_params.create_execution_context_params = pParams;
        module_load_params.internalCommandQueue = false;
        result = npu_vm_load_module(engine, &module_load_params, hRuntime->module);
    }

    if (result != NPU_VM_SUCCESS) {
        npu_vm_destroy_engine(engine);
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    *phExecutionHandle = reinterpret_cast<npu_vm_runtime_execution_context_handle_t>(engine);
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeExecute2(npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_execute_params2_t* pParams) {
    if (hRuntime == nullptr || pParams == nullptr || pParams->executionContext == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    bool recordingRequired = true;
    try {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        const bool useInternalCmdQueue = pParams->commandQueue == nullptr;
        if (useInternalCmdQueue) {
            if (pParams->pConfig) {
                // update configuration of an internal command queue
                configCmdQueue(hRuntime, pParams->pConfig);
            }

            pParams->commandQueue = hRuntime->commandQueue;
        }

        auto engine = reinterpret_cast<npu_vm_engine*>(pParams->executionContext);
        // checks whether the command list needs to be recorded or not.
        // If the command list is already recorded, we can skip the recording step
        // and directly submit the recorded command list for execution.
        recordingRequired = npu_vm_host_code_execution_required(engine, pParams);
        auto result = NPU_VM_RUNTIME_RESULT_SUCCESS;
        if (recordingRequired) {
            if (npu_vm_reset_state(engine, /*resetExecutionContext=*/true) == NPU_VM_SUCCESS) {
                if (npu_vm_infer2(engine, pParams) != NPU_VM_SUCCESS) {
                    NPU_VM_LOG_ERROR("Inference failed for the given parameters");
                    result = NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
                }
            } else {
                NPU_VM_LOG_ERROR("Failed to reset engine state before inference");
                result = NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
            }
        } else {
            // submit recorded command lists
            if (npu_vm_submit_recorded_command_lists(engine, pParams) != NPU_VM_SUCCESS) {
                NPU_VM_LOG_ERROR("Failed to submit recorded command lists for the given parameters");
                result = NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
            }
        }

        if (useInternalCmdQueue) {
            // reset command queue handle
            pParams->commandQueue = nullptr;
        }

        if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            return result;
        }

    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Exception while executing model: {}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    if (recordingRequired) {
        // Reset dirty bits for all input and output memrefs after execution
        // for avoiding unnecessary commandlist recording
        for (uint32_t i = 0; i < pParams->numOfInputs; ++i) {
            auto memRef = reinterpret_cast<npu_vm_runtime_mem_ref*>(pParams->pInputs[i]);
            memRef->resetDirtyBits();
        }

        for (uint32_t i = 0; i < pParams->numOfOutputs; ++i) {
            auto memRef = reinterpret_cast<npu_vm_runtime_mem_ref*>(pParams->pOutputs[i]);
            memRef->resetDirtyBits();
        }
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeHostSync(npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_execute_params2_t* pParams) {
    if (hRuntime == nullptr || pParams == nullptr || pParams->executionContext == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    auto engine = reinterpret_cast<npu_vm_engine*>(pParams->executionContext);

    const auto result = npu_vm_host_sync(engine);
    if (result != NPU_VM_SUCCESS) {
        NPU_VM_LOG_ERROR("Host sync failed for the given parameters: {}", result);
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
