// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef NPU_LLVM_BACKEND

#pragma once 


#include "irgraph.hpp"

#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/runtime/make_tensor.hpp"

#pragma warning(push)
#pragma warning(disable: 4244 4267 4146 4996)
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
#pragma warning(pop)

namespace intel_npu {

#if defined(_WIN32)
#define MLIR_RUNNER_UTILS_FILE_NAME "mlir_runner_utils.dll"
#define MLIR_C_RUNNER_UTILS_FILE_NAME "mlir_c_runner_utils.dll"
#define MLIR_ZERO_WRAPPER_FILE_NAME "level_zero_wrapper.dll"
#else
#define MLIR_RUNNER_UTILS_FILE_NAME "libmlir_runner_utils.so"
#define MLIR_C_RUNNER_UTILS_FILE_NAME "libmlir_c_runner_utils.so"
#define MLIR_ZERO_WRAPPER_FILE_NAME "liblevel_zero_wrapper.so"
#endif

void IRGraph::MemRefType::setArg(const void* arg) {
    basePtr = data = arg;
}

void IRGraph::MemRefType::setSize( const intel_npu::IODescriptor& desc) {
    // Note: check difference between shape from compiler and shape from IR.
    const auto& shape = desc.shapeFromCompiler.get_shape();
    for (size_t i = 0; i < shape.size(); ++i)
        sizes[i] = shape[i];
}

void IRGraph::MemRefType::updateStride() {
    // Note: NCHW layout
    uint64_t stride = 1;
    for (size_t i = 4 - 1; i > 0; --i) {
        strides[i] = stride;
        stride *= sizes[i];
    }
}

IRGraph::GraphArguments::GraphArguments(const GraphArguments& args) {
    *this = args;
}

IRGraph::GraphArguments& IRGraph::GraphArguments::operator=(const GraphArguments& args) {
    if (_inputs.size() != args._inputs.size()) {
        if (_inputs.size() > args._inputs.size()) {
            for (size_t i = args._inputs.size(); i < _inputs.size(); ++i) {
                delete _inputs[i];
                _inputs[i] = nullptr;
            }
        }

        _inputs.resize(args._inputs.size());
    }
    
    auto& inputs = args._inputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (_inputs[i] == nullptr) _inputs[i] = new MemRefType();
        *_inputs[i] = *inputs[i];
    }

    if (_outputs.size() != args._outputs.size()) {
        if (_outputs.size() > args._outputs.size()) {
            for (size_t i = args._outputs.size(); i < _outputs.size(); ++i) {
                delete _outputs[i];
                _outputs[i] = nullptr;
            }
        }
        _outputs.resize(args._outputs.size());
    }

    auto& outputs = args._outputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (_outputs[i] == nullptr) _outputs[i] = new MemRefType();
        *_outputs[i] = *outputs[i];
    }

    return *this;
}

IRGraph::GraphArguments::~GraphArguments() {
    for (auto& input : _inputs) {
        delete input;
    }

    for (auto& output : _outputs) {
        delete output;
    }
}

class IRGraphImpl : public IRGraph::Impl {
public:
    using MemRefType = IRGraph::MemRefType;

public:
    IRGraphImpl() = default;
    void initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata, std::vector<ArgumentDescriptor>& inputs, std::vector<ArgumentDescriptor>& outputs) override;
    std::unique_ptr<mlir::ExecutionEngine> createExecutionEngine(const std::string& entryName, std::optional<ov::Tensor>& blob, mlir::MLIRContext* context);
    void initializeIRGraphExecution(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata, std::vector<ArgumentDescriptor>& inputs, std::vector<ArgumentDescriptor>& outputs);
    void setArgumentValue(uint32_t argi, const void* argv) override;
    void initializeGraph(uint64_t command_queue_group_ordinal) override;
    uint64_t getNumSubgraphs() override { return _numOfSubgraphs; }
    void executeGraph(std::vector<MemRefType*>& inputs, std::vector<MemRefType*>& outputs, const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct, std::vector<ze_command_list_handle_t>& commandLists, ze_command_queue_handle_t commandQueue, ze_fence_handle_t inferenceFence, ze_event_handle_t event, ze_graph_profiling_pool_handle_t profiling);
    void executeGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct, IRGraph::GraphArguments& args, std::vector<ze_command_list_handle_t>& commandLists, ze_command_queue_handle_t commandQueue, ze_fence_handle_t inferenceFence, ze_event_handle_t event, ze_graph_profiling_pool_handle_t profiling);
    void getBinding(IRGraph::GraphArguments& binding);
    virtual ~IRGraphImpl() {}
public:
    std::unique_ptr<mlir::MLIRContext> _context;
    mlir::DialectRegistry _registry;
    std::unique_ptr<mlir::ExecutionEngine> _engine;

    IRGraph::GraphArguments _binding;
    uint64_t _numOfSubgraphs;
    static bool _initializedMLIR;
};

bool IRGraphImpl::_initializedMLIR = false;

void IRGraphImpl::initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata, std::vector<ArgumentDescriptor>& arg_inputs, std::vector<ArgumentDescriptor>& arg_outputs)
{
    if (_initializedMLIR == false) {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        mlir::registerAllToLLVMIRTranslations(_registry);

        _initializedMLIR = true;
    }

    _context = std::make_unique<mlir::MLIRContext>(_registry);
    initializeIRGraphExecution(blob, metadata, arg_inputs, arg_outputs);
    
    _binding._inputs.resize( arg_inputs.size());
   
    auto& inputs = _binding._inputs;
    for (size_t i = 0;i < inputs.size(); ++i) {
        inputs[i] = new MemRefType();
        inputs[i]->setSize(metadata.inputs[i]);
        inputs[i]->updateStride();
    }
    
    _binding._outputs.resize(arg_outputs.size());
    auto& outputs = _binding._outputs;
    for (size_t i = 0;i < outputs.size(); ++i) {
        outputs[i] = new MemRefType();
        outputs[i]->setSize(metadata.outputs[i]);
        outputs[i]->updateStride();
    }
}

std::unique_ptr<mlir::ExecutionEngine> IRGraphImpl::createExecutionEngine(const std::string& entryName, std::optional<ov::Tensor>& blob, mlir::MLIRContext* context) {
    
    auto blobPtr = reinterpret_cast<const uint8_t*>(blob.value().data());
    auto blobSize = blob.value().get_byte_size();
    
    // Metadata<METADATA_VERSION_X_X> is stored after LLVM code in CompiledModel::export_model
    // So, the file size needs to be adjusted to avoid compilation error
    auto getLLVMIRSize = [](const uint8_t* llvmIR, size_t size) {
        if (size == 0 || llvmIR == nullptr) return 0ULL;    
        for (size_t index = size - 1 ; index >= 0; --index) {
            if (llvmIR[index] == static_cast<uint8_t>('}'))
            {
                return index + 1ULL;
            }
        }

        return 0ULL;
    };

    llvm::StringRef content(reinterpret_cast<const char*>(blobPtr), getLLVMIRSize(blobPtr, blobSize));
    auto llvmBlob = llvm::MemoryBuffer::getMemBufferCopy(content, "LLVMBlob");
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(llvmBlob), llvm::SMLoc());
    mlir::OwningOpRef<mlir::Operation*> module = mlir::parseSourceFile<mlir::ModuleOp>(*sourceMgr, context);

    if (!module) {
        OPENVINO_THROW("Failed to parse LLVM IR");
    }

    //std::cout << "Creating JITTargetMachineBuilder" << std::endl;
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
        OPENVINO_THROW("Failed to detect host");
    }
    //std::cout << "Creating TargetMachine for " << tmBuilderOrError->getCPU() << std::endl;
    //std::cout << "Target triple " << tmBuilderOrError->getTargetTriple().normalize() << std::endl;

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
        OPENVINO_THROW("Failed to create TargetMachine");
    }
    //std::cout << "TargetMachine created" << std::endl;

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;

    llvm::SmallVector<mlir::StringRef, 4> sharedLibs;
    sharedLibs.push_back(MLIR_RUNNER_UTILS_FILE_NAME);
    sharedLibs.push_back(MLIR_C_RUNNER_UTILS_FILE_NAME);
    sharedLibs.push_back(MLIR_ZERO_WRAPPER_FILE_NAME);
    engineOptions.sharedLibPaths = sharedLibs;
    engineOptions.enableObjectDump = true;
    //std::cout << "Creating engine" << std::endl;
    auto expectedEngine = mlir::ExecutionEngine::create(*module, engineOptions, std::move(tmOrError.get()));
    if (!expectedEngine) {
        OPENVINO_THROW("Failed to create ExecutionEngine");
    }
    //std::cout << "Engine created" << std::endl;
    auto engine = std::move(*expectedEngine);
    auto expectedFPtr = engine->lookupPacked(entryName);

    if (!expectedFPtr) {
        OPENVINO_THROW("Failed to lookup main function");
    }

    return engine;
}

void IRGraphImpl::getBinding(IRGraph::GraphArguments& binding) {
    binding = _binding;
}

void IRGraphImpl::initializeIRGraphExecution(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata, std::vector<ArgumentDescriptor>& inputs, std::vector<ArgumentDescriptor>& outputs) {
    const std::string adapterPrefix = std::string("_mlir_ciface_");
    const std::string entryName = "main";
    const std::string adapterName = adapterPrefix + entryName;
    _engine = createExecutionEngine(adapterName, blob, _context.get());
    std::string getNetworkMetadataFuncName = "get_network_metadata";
    
    // Get metadata and number of graph
    auto error = _engine->invoke(getNetworkMetadataFuncName, &metadata, &_numOfSubgraphs, &inputs, &outputs);
    if (error) {
        OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
    }

    metadata.bindRelatedDescriptors();
}

void IRGraphImpl::setArgumentValue(uint32_t argi, const void* argv)
{
    auto inputs = _binding._inputs;
    if (argi < inputs.size()) {
        inputs[argi]->basePtr = inputs[argi]->data = const_cast<void*>(argv);
    }
    else {
        auto outputs = _binding._outputs;
        auto idx = argi - inputs.size();
        if (idx < outputs.size()) {
            outputs[idx]->basePtr = outputs[idx]->data = const_cast<void*>(argv);
        }
    }
}

void IRGraphImpl::initializeGraph(uint64_t ordinal) {
    // TODO
}


llvm::Error invokePacked(
    std::unique_ptr<mlir::ExecutionEngine>& engine,
    const std::string& adapterName,
    std::vector<IRGraphImpl::MemRefType*>& inputs,
    std::vector<IRGraphImpl::MemRefType*>& outputs,
    ze_context_handle_t ctx,
    ze_device_handle_t device,
    ze_graph_dditable_ext_t* graphDdiTableExt,
    ze_command_list_handle_t* commandLists,
    uint64_t numCommandLists,
    ze_command_queue_handle_t commandQueue,
    ze_fence_handle_t inferenceFence,
    ze_event_handle_t event)
{
    mlir::SmallVector<void *> packedArgs;
    for (auto& input : inputs) {
        mlir::ExecutionEngine::Argument<IRGraphImpl::MemRefType*>::pack(packedArgs, input);
    }

    for (auto& output : outputs) {
        mlir::ExecutionEngine::Argument<IRGraphImpl::MemRefType*>::pack(packedArgs, output);
    }

    mlir::ExecutionEngine::Argument<ze_context_handle_t>::pack(packedArgs, ctx);
    mlir::ExecutionEngine::Argument<ze_device_handle_t>::pack(packedArgs, device);
    mlir::ExecutionEngine::Argument<ze_graph_dditable_ext_t*>::pack(packedArgs, graphDdiTableExt);
    mlir::ExecutionEngine::Argument<ze_command_list_handle_t*>::pack(packedArgs, commandLists);
    mlir::ExecutionEngine::Argument<uint64_t>::pack(packedArgs, numCommandLists);
    mlir::ExecutionEngine::Argument<ze_command_queue_handle_t>::pack(packedArgs, commandQueue);
    mlir::ExecutionEngine::Argument<ze_fence_handle_t>::pack(packedArgs, inferenceFence);
    mlir::ExecutionEngine::Argument<ze_event_handle_t>::pack(packedArgs, event);

    return engine->invokePacked(adapterName, packedArgs);
}

void IRGraphImpl::executeGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct, IRGraph::GraphArguments& args, std::vector<ze_command_list_handle_t>& commandLists, ze_command_queue_handle_t commandQueue, ze_fence_handle_t fence, ze_event_handle_t event, ze_graph_profiling_pool_handle_t profiling) {
    executeGraph(args._inputs, args._outputs, zeroInitStruct, commandLists, commandQueue, fence, event, profiling);
}

void IRGraphImpl::executeGraph(std::vector<MemRefType*>& inputMefRefs, std::vector<MemRefType*>& outputMemRefs,
    const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct, std::vector<ze_command_list_handle_t>& commandLists, ze_command_queue_handle_t commandQueue, ze_fence_handle_t fence, ze_event_handle_t event, ze_graph_profiling_pool_handle_t) {

    auto contextHandle = zeroInitStruct->getContext();
    auto deviceHandle = zeroInitStruct->getDevice();
    auto ddiTableHandle = zeroInitStruct->getGraphDdiTable().getImpl();
    const std::string adapterName = "_mlir_ciface_main";

    auto error = invokePacked(_engine, adapterName, inputMefRefs, outputMemRefs, contextHandle, deviceHandle,
        ddiTableHandle, commandLists.data(), commandLists.size(), (ze_command_queue_handle_t) commandQueue, fence, event);

    if (error)
        OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
}

IRGraph::IRGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
             std::optional<ov::Tensor> blob,
             bool blobAllocatedByPlugin,
             const Config& config,
             const ov::SoPtr<ICompiler>& compiler)
    : IGraph(config, std::move(blob)),
      _zeroInitStruct(zeroInitStruct),
      _blobAllocatedByPlugin(blobAllocatedByPlugin),
      _compiler(compiler),
      _logger("Graph", config.get<LOG_LEVEL>()) {
    if (!config.get<CREATE_EXECUTOR>() || config.get<DEFER_WEIGHTS_LOAD>()) {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
        return;
    }

    _impl = std::make_unique<IRGraphImpl>();

    // initialize MLIR execution engine, metadata, input&output descriptors
    _impl->initialize(_blob, _metadata, _input_descriptors, _output_descriptors);

    _num_of_subgraphs = _impl->getNumSubgraphs();

    initialize(config);
}

std::pair<uint64_t, std::optional<std::vector<uint64_t>>> IRGraph::export_blob(std::ostream& stream) const {

    const uint8_t* blobPtr = nullptr;
    size_t blobSize = 0;

    std::vector<uint8_t> blobVec;  // plugin needs to keep a copy of the blob for older drivers

    if (_blobIsReleased) {
        OPENVINO_THROW("Model was optimized away. Try importing it using `ov::hint::compiled_blob` property to extend "
                       "its lifetime.");
    }

    if (_blob ==
        std::nullopt) {  // when compiling the model using Compiler in Driver, the blob is handled by the driver
        OPENVINO_THROW("No CiD is supported yet!");
    } else {  // in all other cases, the blob is handled by the plugin
        blobPtr = static_cast<const uint8_t*>(_blob->data());
        blobSize = _blob->get_byte_size();
    }

    if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
        OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
    }
    stream.write(reinterpret_cast<const char*>(blobPtr), static_cast<std::streamsize>(blobSize));

    if (!stream) {
        _logger.error("Write blob to stream failed. Blob is broken!");
        return std::make_pair(0, std::nullopt);
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        std::uint32_t result = 1171117u;
        for (const uint8_t* it = blobPtr; it != blobPtr + blobSize; ++it) {
            result = ((result << 7) + result) + static_cast<uint32_t>(*it);
        }

        std::stringstream str;
        str << "Blob size: " << blobSize << ", hash: " << std::hex << result;
        _logger.info(str.str().c_str());
    }
    _logger.info("Write blob to stream successfully.");

    return std::make_pair(blobSize, std::nullopt);
}

std::vector<ov::ProfilingInfo> IRGraph::process_profiling_output(const std::vector<uint8_t>& profData,
                                                               const Config& config) const {
    if (_compiler == nullptr) {
        OPENVINO_THROW("Profiling post-processing is not supported.");
    }

    std::vector<uint8_t> blob(_blob->get_byte_size());
    blob.assign(reinterpret_cast<const uint8_t*>(_blob->data()),
                reinterpret_cast<const uint8_t*>(_blob->data()) + _blob->get_byte_size());
    return _compiler->process_profiling_output(profData, blob, config);
}

void IRGraph::set_argument_value(uint32_t argi, const void* argv) const {
    if (_impl == nullptr) {
        _logger.warning("Graph handle is null, dynamic pipeline to handle set_argument_value");
        return;
    }

    _impl->setArgumentValue(argi, argv);
}

void IRGraph::initialize(const Config& config) {
    _logger.debug("Graph initialize start");

    if (_command_queue = nullptr) {

        _logger.debug("Graph initialize without graph handle");
        
        _command_queue_group_ordinal =
            zeroUtils::findCommandQueueGroupOrdinal(_zeroInitStruct->getDevice(),
                                                    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

        uint32_t command_queue_options = 0;

        if (config.has<TURBO>() && config.get<TURBO>()) {
            if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 0)) {
                OPENVINO_THROW("Turbo is not supported by the current driver");
            }
            command_queue_options = command_queue_options | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
        }

        if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1) &&
            config.has<RUN_INFERENCES_SEQUENTIALLY>() && config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            command_queue_options = command_queue_options | ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC;
        }

        _command_queue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                        zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                        _command_queue_group_ordinal,
                                                        command_queue_options);

        if (config.has<WORKLOAD_TYPE>()) {
            set_workload_type(config.get<WORKLOAD_TYPE>());
        }

        _impl->initializeGraph(_command_queue_group_ordinal);

        _logger.debug("Graph initialize finish");

        //  We are allowed to release the original blob because weights were loaded in NPU memory during
        //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
        //  releasing it here to avoid unnecessary memory usage.
        //_blobIsReleased = release_blob(config);

        _batch_size = get_batch_size(_metadata);

        if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
            config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            auto number_of_command_lists = _batch_size.has_value() ? *_batch_size : 1;

            _last_submitted_event.resize(number_of_command_lists);
        }
        return;
    }

    _input_descriptors.shrink_to_fit();
    _output_descriptors.shrink_to_fit();

    _command_queue_group_ordinal =
        zeroUtils::findCommandQueueGroupOrdinal(_zeroInitStruct->getDevice(),
                                                ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

    uint32_t command_queue_options = 0;

    if (config.has<TURBO>() && config.get<TURBO>()) {
        if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 0)) {
            _logger.debug("Set ZE_NPU_COMMAND_QUEUE_OPTION_TURBO in command queue options");
            command_queue_options = command_queue_options | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
        }
    }

    if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1) &&
        config.has<RUN_INFERENCES_SEQUENTIALLY>() && config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        _logger.debug("Set ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC in command queue options");
        command_queue_options = command_queue_options | ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC;
    }

    _command_queue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                    zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                    _command_queue_group_ordinal,
                                                    command_queue_options);

    if (config.has<WORKLOAD_TYPE>()) {
        set_workload_type(config.get<WORKLOAD_TYPE>());
    }

    // TODO
    // invoke for graph intialization 
    // engine->invoke("initialization")

    _logger.debug("Graph initialize finish");

    //  We are allowed to release the original blob because weights were loaded in NPU memory during
    //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
    //  releasing it here to avoid unnecessary memory usage.
    _blobIsReleased = release_blob(config);

    _batch_size = get_batch_size(_metadata);

    if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
        config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        auto number_of_command_lists = _batch_size.has_value() ? *_batch_size : 1;

        _last_submitted_event.resize(number_of_command_lists);
    }
}

bool IRGraph::release_blob(const Config& config) {
    if (!_blobAllocatedByPlugin) {
        return false;
    }

    if (_blob == std::nullopt || _zeroInitStruct->getGraphDdiTable().version() < ZE_GRAPH_EXT_VERSION_1_8 ||
        config.get<PERF_COUNT>()) {
        return false;
    }

    ze_graph_properties_2_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    _zeroInitStruct->getGraphDdiTable().pfnGetProperties2(_handle, &properties);

    if (~properties.initStageRequired & ZE_GRAPH_STAGE_INITIALIZE) {
        return false;
    }

    _blob = std::nullopt;
    _logger.debug("Blob is released");

    return true;
};

IRGraph::~IRGraph() {
    // make sure all the context-dependent components are destroyed before the zero context is destroyed
    if (_handle != nullptr) {
        _handle = nullptr;
    }

    if (!_last_submitted_event.empty()) {
        _last_submitted_event.clear();
    }

    if (_command_queue != nullptr) {
        _command_queue.reset();
    }
}

void IRGraph::execute(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct, IRGraph::GraphArguments& args, std::vector<ze_command_list_handle_t>& commandLists, ze_command_queue_handle_t commandQueue, ze_fence_handle_t inferenceFence, ze_event_handle_t event, ze_graph_profiling_pool_handle_t profiling)
{
    auto impl = reinterpret_cast<IRGraphImpl*>(_impl.get());

    if (impl == nullptr) return;

    impl->executeGraph(zeroInitStruct, args, commandLists, commandQueue, inferenceFence, event, profiling);
}

void IRGraph::getBinding( GraphArguments& args )
{
    auto impl = reinterpret_cast<IRGraphImpl*>(_impl.get());
    
    if (impl == nullptr) return;
    
    impl->getBinding(args);
}
}  // namespace intel_npu
#endif // NPU_LLVM_BACKEND