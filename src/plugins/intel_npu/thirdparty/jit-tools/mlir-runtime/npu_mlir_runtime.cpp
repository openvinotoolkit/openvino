// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include "intel_npu/config/options.hpp"
// #include "intel_npu/utils/utils.hpp"
// #include "intel_npu/utils/zero/zero_api.hpp"
// #include "openvino/runtime/make_tensor.hpp"
#include "npu_mlir_runtime.hpp"

#include "intel_npu/icompiler.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

#if defined(_WIN32)
#    pragma warning(push)
#    pragma warning(disable : 4244 4267 4146 4996)
#endif
#include <llvm/Support/Error.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#if defined(_WIN32)
#    pragma warning(pop)
#endif

using namespace intel_npu;

#if defined(_WIN32)
//#    define MLIR_RUNNER_UTILS_FILE_NAME   "mlir_runner_utils.dll"
//#    define MLIR_C_RUNNER_UTILS_FILE_NAME "mlir_c_runner_utils.dll"
#    define MLIR_ZERO_WRAPPER_FILE_NAME "level_zero_wrapper.dll"
#else
//#    define MLIR_RUNNER_UTILS_FILE_NAME   "libmlir_runner_utils.so"
//#    define MLIR_C_RUNNER_UTILS_FILE_NAME "libmlir_c_runner_utils.so"
#    define MLIR_ZERO_WRAPPER_FILE_NAME   "liblevel_zero_wrapper.so"
#endif

class NPUMLIRRuntime {
public:
    NPUMLIRRuntime(const npu_mlir_runtime_blob_desc_t* desc, npu_mlir_runtime_properties_t* pProperties);
    ~NPUMLIRRuntime();

    void createExecutionEngine(const npu_mlir_runtime_blob_desc_t* blob);

    void parseMetadata();

    void getArgumentProperties(uint32_t argIndex,
                               ze_graph_argument_properties_3_t* pGraphArgumentProperties,
                               ze_graph_argument_metadata_t* pGraphArgumentMetadata);

    void execute(npu_mlir_runtime_execute_params_t* pParams);

private:
    std::unique_ptr<mlir::MLIRContext> _context;
    mlir::DialectRegistry _registry;
    std::unique_ptr<mlir::ExecutionEngine> _engine;

    NetworkMetadata _metadata;
    std::vector<ArgumentDescriptor> _inputs;
    std::vector<ArgumentDescriptor> _outputs;
    uint32_t _numOfSubgraphs = 0;
    uint32_t _numOfArgs = 0;
    Logger _logger = Logger("NPUMLIRRuntime", Logger::global().level());
};

void NPUMLIRRuntime::createExecutionEngine(const npu_mlir_runtime_blob_desc_t* desc) {
    _logger.debug("Creating execution engine from blob at %p of size %zu", desc->pInput, desc->inputSize);
    const std::string adapterPrefix = std::string("_mlir_ciface_");
    const std::string entryName = "main";
    const std::string adapterName = adapterPrefix + entryName;

    auto blobPtr = desc->pInput;
    auto blobSize = desc->inputSize;

    // Metadata<METADATA_VERSION_X_X> is stored after LLVM code in CompiledModel::export_model
    // So, the file size needs to be adjusted to avoid compilation error
    auto getLLVMIRSize = [](const uint8_t* llvmIR, size_t size) {
        if (size == 0 || llvmIR == nullptr)
            return 0ULL;
        for (size_t index = size - 1; index >= 0; --index) {
            if (llvmIR[index] == static_cast<uint8_t>('}')) {
                return index + 1ULL;
            }
        }

        return 0ULL;
    };

    llvm::StringRef content(reinterpret_cast<const char*>(blobPtr), getLLVMIRSize(blobPtr, blobSize));
    auto llvmBlob = llvm::MemoryBuffer::getMemBufferCopy(content, "LLVMBlob");
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(llvmBlob), llvm::SMLoc());
    mlir::OwningOpRef<mlir::Operation*> module = mlir::parseSourceFile<mlir::ModuleOp>(*sourceMgr, _context.get());

    if (!module) {
        OPENVINO_THROW("Failed to parse LLVM IR");
    }

    // std::cout << "Creating JITTargetMachineBuilder" << std::endl;
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
        OPENVINO_THROW("Failed to detect host");
    }
    // std::cout << "Creating TargetMachine for " << tmBuilderOrError->getCPU() << std::endl;
    // std::cout << "Target triple " << tmBuilderOrError->getTargetTriple().normalize() << std::endl;

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
        OPENVINO_THROW("Failed to create TargetMachine");
    }
    // std::cout << "TargetMachine created" << std::endl;

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;

    llvm::SmallVector<mlir::StringRef, 4> sharedLibs;
    // sharedLibs.push_back(MLIR_RUNNER_UTILS_FILE_NAME);
    // sharedLibs.push_back(MLIR_C_RUNNER_UTILS_FILE_NAME);
    sharedLibs.push_back(MLIR_ZERO_WRAPPER_FILE_NAME);
    engineOptions.sharedLibPaths = sharedLibs;
    // std::cout << "Creating engine" << std::endl;
    auto expectedEngine = mlir::ExecutionEngine::create(*module, engineOptions, std::move(tmOrError.get()));
    if (!expectedEngine) {
        OPENVINO_THROW("Failed to create ExecutionEngine");
    }
    // std::cout << "Engine created" << std::endl;
    _engine = std::move(*expectedEngine);
    auto expectedFPtr = _engine->lookupPacked(entryName);

    if (!expectedFPtr) {
        OPENVINO_THROW("Failed to lookup main function");
    }
}

void NPUMLIRRuntime::parseMetadata() {
    _logger.debug("Parsing metadata");
    std::string getNetworkMetadataFuncName = "get_network_metadata";

    // Get metadata and number of graph
    auto error = _engine->invoke(getNetworkMetadataFuncName, &_metadata, &_numOfSubgraphs, &_inputs, &_outputs);
    if (error) {
        OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
    }
    _logger.debug("num of subgraphs: %d inputs: %d outputs: %d", _numOfSubgraphs, _inputs.size(), _outputs.size());
    _metadata.bindRelatedDescriptors();
    _numOfArgs = _inputs.size() + _outputs.size();
}

NPUMLIRRuntime::NPUMLIRRuntime(const npu_mlir_runtime_blob_desc_t* desc, npu_mlir_runtime_properties_t* pProperties) {
    // Initialize MLIR context and register necessary dialects
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    mlir::registerAllToLLVMIRTranslations(_registry);

    _context = std::make_unique<mlir::MLIRContext>(_registry);

    createExecutionEngine(desc);

    parseMetadata();

    pProperties->numOfSubGraphs = _numOfSubgraphs;
    pProperties->numOfGraphArgs = _numOfArgs;
}

NPUMLIRRuntime::~NPUMLIRRuntime() {
    _engine.reset();
    _context.reset();
}

void NPUMLIRRuntime::getArgumentProperties(uint32_t argIndex,
                                           ze_graph_argument_properties_3_t* pGraphArgumentProperties,
                                           ze_graph_argument_metadata_t* pGraphArgumentMetadata) {
    _logger.debug("Getting argument properties for index %d", argIndex);
    if (argIndex >= _numOfArgs) {
        OPENVINO_THROW("Invalid argument index");
    }

    const ArgumentDescriptor* argDesc = nullptr;
    IODescriptor desc;
    if (argIndex < _inputs.size()) {
        argDesc = &_inputs[argIndex];
        desc = _metadata.inputs[argIndex];
    } else {
        argDesc = &_outputs[argIndex - _inputs.size()];
        desc = _metadata.outputs[argIndex - _inputs.size()];
    }

    // Define new struct to hold metadata
    *pGraphArgumentProperties = argDesc->info;

    // Fill in metadata struct
    pGraphArgumentMetadata->stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_METADATA;
    pGraphArgumentMetadata->pNext = nullptr;
    pGraphArgumentMetadata->type = argDesc->info.type;
    std::strncpy(pGraphArgumentMetadata->friendly_name, argDesc->info.name, ZE_MAX_GRAPH_ARGUMENT_NAME);
    pGraphArgumentMetadata->data_type = ZE_GRAPH_METADATA_TYPE_UNDEFINED;

    if (desc.shapeFromIRModel.has_value()) {
        // Only care about shape, this is shapeFromIRModel
        for (size_t i = 0; i < desc.shapeFromIRModel->size() && i < ZE_MAX_GRAPH_TENSOR_REF_DIMS; ++i) {
            auto val = desc.shapeFromIRModel.value()[i];
            pGraphArgumentMetadata->shape[i] =
                val.is_dynamic() ? std::numeric_limits<uint64_t>::max() : val.get_length();
        }
    } else {
        // Use shapeFromCompiler
        std::copy(std::begin(desc.shapeFromCompiler.get_shape()),
                  std::end(desc.shapeFromCompiler.get_shape()),
                  std::begin(pGraphArgumentMetadata->shape));
    }
    pGraphArgumentMetadata->shape_size = argDesc->info.dims_count;
    pGraphArgumentMetadata->tensor_names_count = 0;  // Not used
    std::strncpy(pGraphArgumentMetadata->input_name, argDesc->info.name, ZE_MAX_GRAPH_ARGUMENT_NAME);
}

void NPUMLIRRuntime::execute(npu_mlir_runtime_execute_params_t* pParams) {
    _logger.debug("Executing with %d inputs and %d outputs", pParams->numOfInputs, pParams->numOfOutputs);
    if (pParams == nullptr) {
        OPENVINO_THROW("Invalid execute parameters");
    }

    mlir::SmallVector<void*> packedArgs;
    for (uint32_t i = 0; i < pParams->numOfInputs; ++i) {
        mlir::ExecutionEngine::Argument<npu_mlir_runtime_mem_ref_t*>::pack(packedArgs, pParams->pInputs[i]);
    }

    for (uint32_t i = 0; i < pParams->numOfOutputs; ++i) {
        mlir::ExecutionEngine::Argument<npu_mlir_runtime_mem_ref_t*>::pack(packedArgs, pParams->pOutputs[i]);
    }

    mlir::ExecutionEngine::Argument<ze_context_handle_t>::pack(packedArgs, pParams->ctx);
    mlir::ExecutionEngine::Argument<ze_device_handle_t>::pack(packedArgs, pParams->device);
    mlir::ExecutionEngine::Argument<ze_graph_dditable_ext_t*>::pack(packedArgs, pParams->graphDdiTableExt);
    mlir::ExecutionEngine::Argument<ze_command_list_handle_t*>::pack(packedArgs, pParams->commandLists);
    mlir::ExecutionEngine::Argument<uint64_t>::pack(packedArgs, pParams->numCommandLists);
    mlir::ExecutionEngine::Argument<ze_command_queue_handle_t>::pack(packedArgs, pParams->commandQueue);
    mlir::ExecutionEngine::Argument<ze_fence_handle_t>::pack(packedArgs, pParams->inferenceFence);
    mlir::ExecutionEngine::Argument<ze_event_handle_t>::pack(packedArgs, pParams->event);

    const std::string adapterName = "_mlir_ciface_main";
    auto error = _engine->invokePacked(adapterName, packedArgs);
    if (error)
        OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
}

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#    define DLLEXPORT __declspec(dllexport)
#else
#    define DLLEXPORT __attribute__((visibility("default")))
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Init MLIR runtime instance and return handle
DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeCreate(const npu_mlir_runtime_blob_desc_t* desc,
                     npu_mlir_runtime_handle_t* phRuntime,
                     npu_mlir_runtime_properties_t* pProperties) {
    if (phRuntime == nullptr || desc == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = new NPUMLIRRuntime(desc, pProperties);
        *phRuntime = reinterpret_cast<npu_mlir_runtime_handle_t>(runtime);
    } catch (const std::exception& e) {
        printf("Error creating MLIR runtime: %s", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy MLIR runtime instance
DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL npuMLIRRuntimeDestroy(npu_mlir_runtime_handle_t hRuntime) {
    if (hRuntime == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        delete runtime;
    } catch (const std::exception& e) {
        printf("Error destroying MLIR runtime: %s", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get metadata from MLIR runtime instance
DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeGetMetadata(npu_mlir_runtime_handle_t hRuntime,
                          uint32_t argIndex,
                          ze_graph_argument_properties_3_t* pGraphArgumentProperties,
                          ze_graph_argument_metadata_t* pGraphArgumentMetadata) {
    if (hRuntime == nullptr || pGraphArgumentProperties == nullptr || pGraphArgumentMetadata == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        runtime->getArgumentProperties(argIndex, pGraphArgumentProperties, pGraphArgumentMetadata);
    } catch (const std::exception& e) {
        printf("Error getting argument properties: %s", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeExecute(npu_mlir_runtime_handle_t hRuntime, npu_mlir_runtime_execute_params_t* pParams) {
    if (hRuntime == nullptr || pParams == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        runtime->execute(pParams);
    } catch (const std::exception& e) {
        printf("Error executing MLIR runtime: %s", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

#ifdef __cplusplus
}
#endif