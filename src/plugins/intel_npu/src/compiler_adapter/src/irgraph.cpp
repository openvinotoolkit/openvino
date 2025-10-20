// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#    pragma once

#    include "irgraph.hpp"

#    include "intel_npu/config/options.hpp"
#    include "intel_npu/utils/utils.hpp"
#    include "intel_npu/utils/zero/zero_api.hpp"
#    include "openvino/runtime/make_tensor.hpp"

#    pragma warning(push)
#    pragma warning(disable : 4244 4267 4146 4996)
#    include <llvm/Support/Error.h>
#    include <llvm/Support/InitLLVM.h>
#    include <llvm/Support/SourceMgr.h>
#    include <llvm/Support/TargetSelect.h>
#    include <mlir/ExecutionEngine/ExecutionEngine.h>
#    include <mlir/ExecutionEngine/MemRefUtils.h>
#    include <mlir/IR/BuiltinOps.h>
#    include <mlir/IR/DialectRegistry.h>
#    include <mlir/IR/MLIRContext.h>
#    include <mlir/Parser/Parser.h>
#    include <mlir/Support/LLVM.h>
#    include <mlir/Target/LLVMIR/Dialect/All.h>
#    pragma warning(pop)

namespace intel_npu {

#    if defined(_WIN32)
#        define MLIR_RUNNER_UTILS_FILE_NAME   "mlir_runner_utils.dll"
#        define MLIR_C_RUNNER_UTILS_FILE_NAME "mlir_c_runner_utils.dll"
#        define MLIR_ZERO_WRAPPER_FILE_NAME   "level_zero_wrapper.dll"
#    else
#        define MLIR_RUNNER_UTILS_FILE_NAME   "libmlir_runner_utils.so"
#        define MLIR_C_RUNNER_UTILS_FILE_NAME "libmlir_c_runner_utils.so"
#        define MLIR_ZERO_WRAPPER_FILE_NAME   "liblevel_zero_wrapper.so"
#    endif

void IRGraph::MemRefType::setArg(const void* arg) {
    basePtr = data = arg;
}

void IRGraph::MemRefType::setSize(const intel_npu::IODescriptor& desc) {
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
        if (_inputs[i] == nullptr)
            _inputs[i] = new MemRefType();
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
        if (_outputs[i] == nullptr)
            _outputs[i] = new MemRefType();
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
    IRGraphImpl() : _logger("IRGraphImpl", Logger::global().level()) {}
    void initialize(std::optional<ov::Tensor>& blob,
                    NetworkMetadata& metadata,
                    std::vector<ArgumentDescriptor>& inputs,
                    std::vector<ArgumentDescriptor>& outputs) override;
    std::unique_ptr<mlir::ExecutionEngine> createExecutionEngine(const std::string& entryName,
                                                                 std::optional<ov::Tensor>& blob,
                                                                 mlir::MLIRContext* context);
    void initializeIRGraphExecution(std::optional<ov::Tensor>& blob,
                                    NetworkMetadata& metadata,
                                    std::vector<ArgumentDescriptor>& inputs,
                                    std::vector<ArgumentDescriptor>& outputs);
    void setArgumentValue(uint32_t argi, const void* argv) override;
    void setArgumentProperty(uint32_t argi,
                             const void* argv,
                             const ov::Strides strides,
                             const ov::Shape& shapes) override;
    void initializeGraph(uint64_t command_queue_group_ordinal) override;
    uint64_t getNumSubgraphs() override {
        return _numOfSubgraphs;
    }
    void executeGraph(std::vector<MemRefType*>& inputs,
                      std::vector<MemRefType*>& outputs,
                      const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                      std::vector<ze_command_list_handle_t>& commandLists,
                      ze_command_queue_handle_t commandQueue,
                      ze_fence_handle_t inferenceFence,
                      ze_event_handle_t event,
                      ze_graph_profiling_pool_handle_t profiling);
    void executeGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                      IRGraph::GraphArguments& args,
                      std::vector<ze_command_list_handle_t>& commandLists,
                      ze_command_queue_handle_t commandQueue,
                      ze_fence_handle_t inferenceFence,
                      ze_event_handle_t event,
                      ze_graph_profiling_pool_handle_t profiling) override;
    void getBinding(IRGraph::GraphArguments& binding) override;
    virtual ~IRGraphImpl() {}

public:
    std::unique_ptr<mlir::MLIRContext> _context;
    mlir::DialectRegistry _registry;
    std::unique_ptr<mlir::ExecutionEngine> _engine;

    IRGraph::GraphArguments _binding;
    uint64_t _numOfSubgraphs;
    static bool _initializedMLIR;
    Logger _logger;
};

bool IRGraphImpl::_initializedMLIR = false;

void IRGraphImpl::initialize(std::optional<ov::Tensor>& blob,
                             NetworkMetadata& metadata,
                             std::vector<ArgumentDescriptor>& arg_inputs,
                             std::vector<ArgumentDescriptor>& arg_outputs) {
    if (_initializedMLIR == false) {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        mlir::registerAllToLLVMIRTranslations(_registry);

        // Need to call initialize each time, otherwise the engine can not find llvm dialect
        //_initializedMLIR = true;
    }

    _context = std::make_unique<mlir::MLIRContext>(_registry);
    initializeIRGraphExecution(blob, metadata, arg_inputs, arg_outputs);

    _binding._inputs.resize(arg_inputs.size());

    // dump output of _metadata
    _logger.debug("Dump metadata info from blob");
    _logger.debug("Metadata inputs: %d", metadata.inputs.size());
    for (const auto& input : metadata.inputs) {
        _logger.debug("Input compiler name: %s input node name: %s shape: %s",
                      input.nameFromCompiler.c_str(),
                      input.nodeFriendlyName.c_str(),
                      input.shapeFromCompiler.to_string().c_str());
    }
    _logger.debug("Metadata outputs: %d", metadata.outputs.size());
    for (const auto& output : metadata.outputs) {
        _logger.debug("Output compiler name: %s output node name: %s shape: %s",
                      output.nameFromCompiler.c_str(),
                      output.nodeFriendlyName.c_str(),
                      output.shapeFromCompiler.to_string().c_str());
    }

    _logger.debug("Dump MemRefType from initial metadata:");
    _logger.debug("Inputs:");
    auto& inputs = _binding._inputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = new MemRefType();
        inputs[i]->setSize(metadata.inputs[i]);
        inputs[i]->updateStride();
        std::ostringstream oss;
        oss << (*inputs[i]);
        _logger.debug("MemRefType for input %d : %s", i, oss.str().c_str());
    }

    _logger.debug("Outputs:");
    _binding._outputs.resize(arg_outputs.size());
    auto& outputs = _binding._outputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i] = new MemRefType();
        outputs[i]->setSize(metadata.outputs[i]);
        outputs[i]->updateStride();
        std::ostringstream oss;
        oss << (*outputs[i]);
        _logger.debug("MemRefType for output %d : %s", i, oss.str().c_str());
    }
}

std::unique_ptr<mlir::ExecutionEngine> IRGraphImpl::createExecutionEngine(const std::string& entryName,
                                                                          std::optional<ov::Tensor>& blob,
                                                                          mlir::MLIRContext* context) {
    auto blobPtr = reinterpret_cast<const uint8_t*>(blob.value().data());
    auto blobSize = blob.value().get_byte_size();

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
    mlir::OwningOpRef<mlir::Operation*> module = mlir::parseSourceFile<mlir::ModuleOp>(*sourceMgr, context);

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
    sharedLibs.push_back(MLIR_RUNNER_UTILS_FILE_NAME);
    sharedLibs.push_back(MLIR_C_RUNNER_UTILS_FILE_NAME);
    sharedLibs.push_back(MLIR_ZERO_WRAPPER_FILE_NAME);
    engineOptions.sharedLibPaths = sharedLibs;
    engineOptions.enableObjectDump = true;
    // std::cout << "Creating engine" << std::endl;
    auto expectedEngine = mlir::ExecutionEngine::create(*module, engineOptions, std::move(tmOrError.get()));
    if (!expectedEngine) {
        OPENVINO_THROW("Failed to create ExecutionEngine");
    }
    // std::cout << "Engine created" << std::endl;
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

void IRGraphImpl::initializeIRGraphExecution(std::optional<ov::Tensor>& blob,
                                             NetworkMetadata& metadata,
                                             std::vector<ArgumentDescriptor>& inputs,
                                             std::vector<ArgumentDescriptor>& outputs) {
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
    _logger.debug("num of subgraphs: %d inputs: %d outputs: %d", _numOfSubgraphs, inputs.size(), outputs.size());

    metadata.bindRelatedDescriptors();
}

void IRGraphImpl::setArgumentValue(uint32_t argi, const void* argv) {
    auto inputs = _binding._inputs;
    if (argi < inputs.size()) {
        _logger.debug("setArgumentValue for index %d (input %d)", argi, argi);
        inputs[argi]->basePtr = inputs[argi]->data = const_cast<void*>(argv);
    } else {
        auto outputs = _binding._outputs;
        auto idx = argi - inputs.size();
        _logger.debug("setArgumentValue for index %d (output %d)", argi, idx);
        if (idx < outputs.size()) {
            outputs[idx]->basePtr = outputs[idx]->data = const_cast<void*>(argv);
        }
    }
}

void IRGraphImpl::setArgumentProperty(uint32_t argi,
                                      const void* argv,
                                      const ov::Strides strides,
                                      const ov::Shape& shapes) {
    _logger.debug("setArgumentProperty for index %d", argi);
    auto inputs = _binding._inputs;
    if (argi < inputs.size()) {
        std::ostringstream oss;
        oss << *(inputs[argi]);
        _logger.debug("setArgumentProperty for index %d (input %d)", argi, argi);
        _logger.debug("Before change: %s", oss.str().c_str());
        inputs[argi]->basePtr = inputs[argi]->data = const_cast<void*>(argv);
        // Now MemRefType only support 4 dimension
        size_t shapesSize = shapes.size();
        for (size_t i = 0; i < 4; i++) {
            if (i < shapesSize) {
                inputs[argi]->sizes[i] = shapes[i];
            } else {
                // Set dimension to 1 if exceed region of shapes
                inputs[argi]->sizes[i] = 1;
            }
        }

        size_t stridesSize = strides.size();
        for (size_t i = 0; i < 4; i++) {
            if (i < stridesSize) {
                inputs[argi]->strides[i] = strides[i];
            } else {
                // Set dimension to 1 if exceed region of shapes
                inputs[argi]->strides[i] = 1;
            }
        }

        // Need stride based on element but not byte
        inputs[argi]->updateStride();
        oss.clear();
        oss.str("");
        oss << *(inputs[argi]);
        _logger.debug("After change: %s", oss.str().c_str());

    } else {
        auto outputs = _binding._outputs;
        auto idx = argi - inputs.size();
        _logger.debug("setArgumentValue for index %d (output %d)", argi, idx);
        if (idx < outputs.size()) {
            std::ostringstream oss;
            oss << *(outputs[idx]);
            _logger.debug("Before change: %s", oss.str().c_str());
            outputs[idx]->basePtr = outputs[idx]->data = const_cast<void*>(argv);

            // Now MemRefType only support 4 dimension
            size_t shapesSize = shapes.size();
            for (size_t i = 0; i < 4; i++) {
                if (i < shapesSize) {
                    outputs[idx]->sizes[i] = shapes[i];
                } else {
                    // Set dimension to 1 if exceed region of shapes
                    outputs[idx]->sizes[i] = 1;
                }
            }

            size_t stridesSize = strides.size();
            for (size_t i = 0; i < 4; i++) {
                if (i < stridesSize) {
                    outputs[idx]->strides[i] = strides[i];
                } else {
                    // Set dimension to 1 if exceed region of shapes
                    outputs[idx]->strides[i] = 1;
                }
            }

            // Need stride based on element but not byte
            outputs[idx]->updateStride();

            oss.clear();
            oss.str("");
            oss << *(outputs[idx]);
            _logger.debug("After change: %s", oss.str().c_str());
        }
    }
}

void IRGraphImpl::initializeGraph(uint64_t ordinal) {
    // TODO
}

llvm::Error invokePacked(std::unique_ptr<mlir::ExecutionEngine>& engine,
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
                         ze_event_handle_t event) {
    mlir::SmallVector<void*> packedArgs;
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

void IRGraphImpl::executeGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                               IRGraph::GraphArguments& args,
                               std::vector<ze_command_list_handle_t>& commandLists,
                               ze_command_queue_handle_t commandQueue,
                               ze_fence_handle_t fence,
                               ze_event_handle_t event,
                               ze_graph_profiling_pool_handle_t profiling) {
    executeGraph(args._inputs, args._outputs, zeroInitStruct, commandLists, commandQueue, fence, event, profiling);
}

void IRGraphImpl::executeGraph(std::vector<MemRefType*>& inputMefRefs,
                               std::vector<MemRefType*>& outputMemRefs,
                               const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                               std::vector<ze_command_list_handle_t>& commandLists,
                               ze_command_queue_handle_t commandQueue,
                               ze_fence_handle_t fence,
                               ze_event_handle_t event,
                               ze_graph_profiling_pool_handle_t) {
    auto contextHandle = zeroInitStruct->getContext();
    auto deviceHandle = zeroInitStruct->getDevice();
    auto ddiTableHandle = zeroInitStruct->getGraphDdiTable().getImpl();
    const std::string adapterName = "_mlir_ciface_main";

    auto error = invokePacked(_engine,
                              adapterName,
                              inputMefRefs,
                              outputMemRefs,
                              contextHandle,
                              deviceHandle,
                              ddiTableHandle,
                              commandLists.data(),
                              commandLists.size(),
                              (ze_command_queue_handle_t)commandQueue,
                              fence,
                              event);

    if (error)
        OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
}

IRGraph::IRGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                 std::optional<ov::Tensor> blob,
                 bool blobAllocatedByPlugin,
                 const Config& config,
                 const ov::SoPtr<ICompiler>& compiler)
    : _zeroInitStruct(zeroInitStruct),
      _blob(std::move(blob)),
      _blobAllocatedByPlugin(blobAllocatedByPlugin),
      _compiler(compiler),
      _logger("Graph", config.get<LOG_LEVEL>()) {
    if (!config.get<CREATE_EXECUTOR>() || config.get<DEFER_WEIGHTS_LOAD>()) {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
        return;
    }

    _impl = std::make_unique<IRGraphImpl>();

    // initialize MLIR execution engine, metadata, input&output descriptors
    _impl->initialize(_blob, _metadata, _inputDescriptors, _outputDescriptors);

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

    size_t size = utils::align_size_to_standard_page_size(blobSize);
    size_t paddingSize = size - blobSize;
    if (paddingSize > 0) {
        std::fill_n(std::ostream_iterator<char>(stream), paddingSize, 0);
        if (!stream) {
            _logger.error("Write padding to stream failed. Blob is broken!");
            return std::make_pair(0, std::nullopt);
        }
        _logger.info("Blob size with padding: %ld", size);
    }
    _logger.info("Write blob to stream successfully.");
    return std::make_pair(size, std::nullopt);
}

const NetworkMetadata& IRGraph::get_metadata() const {
    return _metadata;
}

void IRGraph::update_network_name(std::string_view name) {
    _metadata.name = name;
}

const std::vector<ArgumentDescriptor>& IRGraph::get_input_descriptors() const {
    return _inputDescriptors;
}

const std::vector<ArgumentDescriptor>& IRGraph::get_output_descriptors() const {
    return _outputDescriptors;
}

const std::shared_ptr<CommandQueue>& IRGraph::get_command_queue() const {
    return _commandQueue;
}

uint32_t IRGraph::get_command_queue_group_ordinal() const {
    return _commandQueueGroupOrdinal;
}

void IRGraph::set_workload_type(const ov::WorkloadType workloadType) const {
    if (_commandQueue == nullptr) {
        return;
    }

    ze_command_queue_workload_type_t zeWorkloadType;
    switch (workloadType) {
    case ov::WorkloadType::DEFAULT:
        zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_DEFAULT;
        break;
    case ov::WorkloadType::EFFICIENT:
        zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_BACKGROUND;
        break;
    default:
        OPENVINO_THROW("Unknown value for WorkloadType!");
    }

    _commandQueue->setWorkloadType(zeWorkloadType);
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

ze_graph_handle_t IRGraph::get_handle() const {
    _logger.warning("IRGraph does not support get_handle() method.");
    return nullptr;
}

void IRGraph::set_argument_property(uint32_t argi,
                                    const void* argv,
                                    const ov::Strides& strides,
                                    const ov::Shape& shapes) const {
    if (_impl == nullptr) {
        _logger.warning("Graph handle is null, dynamic pipeline to handle set_argument_value");
        return;
    }

    _impl->setArgumentProperty(argi, argv, strides, shapes);
}

void IRGraph::initialize(const Config& config) {
    _logger.debug("Graph initialize start");

    if (_commandQueue = nullptr) {
        _logger.debug("Graph initialize without graph handle");

        _commandQueueGroupOrdinal =
            zeroUtils::findCommandQueueGroupOrdinal(_zeroInitStruct->getDevice(),
                                                    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

        uint32_t commandQueueOptions = 0;

        if (config.has<TURBO>() && config.get<TURBO>()) {
            if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 0)) {
                OPENVINO_THROW("Turbo is not supported by the current driver");
            }
            commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
        }

        if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1) &&
            config.has<RUN_INFERENCES_SEQUENTIALLY>() && config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC;
        }

        _commandQueue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                       zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                       _commandQueueGroupOrdinal,
                                                       commandQueueOptions);

        if (config.has<WORKLOAD_TYPE>()) {
            set_workload_type(config.get<WORKLOAD_TYPE>());
        }

        _impl->initializeGraph(_commandQueueGroupOrdinal);

        _logger.debug("Graph initialize finish");

        //  We are allowed to release the original blob because weights were loaded in NPU memory during
        //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
        //  releasing it here to avoid unnecessary memory usage.
        //_blobIsReleased = release_blob(config);

        _batchSize = determine_batch_size();

        if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
            config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            auto numberOfCommandLists = _batchSize.has_value() ? *_batchSize : 1;

            _lastSubmittedEvent.resize(numberOfCommandLists);
        }
        return;
    }

    _inputDescriptors.shrink_to_fit();
    _outputDescriptors.shrink_to_fit();

    _commandQueueGroupOrdinal = zeroUtils::findCommandQueueGroupOrdinal(_zeroInitStruct->getDevice(),
                                                                        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

    uint32_t commandQueueOptions = 0;

    if (config.has<TURBO>() && config.get<TURBO>()) {
        if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 0)) {
            _logger.debug("Set ZE_NPU_COMMAND_QUEUE_OPTION_TURBO in command queue options");
            commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
        }
    }

    if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1) &&
        config.has<RUN_INFERENCES_SEQUENTIALLY>() && config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        _logger.debug("Set ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC in command queue options");
        commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC;
    }

    _commandQueue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                   zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                   _commandQueueGroupOrdinal,
                                                   commandQueueOptions);

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

    _batchSize = determine_batch_size();

    if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
        config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        auto numberOfCommandLists = _batchSize.has_value() ? *_batchSize : 1;

        _lastSubmittedEvent.resize(numberOfCommandLists);
    }
}

bool IRGraph::release_blob(const Config& config) {
    _logger.warning("Release blob is skipped, no handle for IRGraph");
    // if (!_blobAllocatedByPlugin) {
    //     return false;
    // }

    // if (_blob == std::nullopt || _zeroInitStruct->getGraphDdiTable().version() < ZE_GRAPH_EXT_VERSION_1_8 ||
    //     config.get<PERF_COUNT>()) {
    //     return false;
    // }

    // ze_graph_properties_2_t properties = {};
    // properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    // _zeroInitStruct->getGraphDdiTable().pfnGetProperties2(_handle, &properties);

    // if (~properties.initStageRequired & ZE_GRAPH_STAGE_INITIALIZE) {
    //     return false;
    // }

    // _blob = std::nullopt;
    // _logger.debug("Blob is released");

    return false;
};

void IRGraph::set_last_submitted_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList) {
    _lastSubmittedEvent[indexOfCommandList] = event;
}

const std::shared_ptr<Event>& IRGraph::get_last_submitted_event(size_t indexOfCommandList) const {
    return _lastSubmittedEvent[indexOfCommandList];
}

void IRGraph::resize_last_submitted_event(size_t batch) {
    _lastSubmittedEvent.resize(batch);
}

void IRGraph::set_batch_size(std::size_t batch) {
    _batchSize = batch;
}

uint32_t IRGraph::get_unique_id() {
    return _uniqueId++;
}

void IRGraph::set_last_submitted_id(uint32_t id_index) {
    _lastSubmittedId = id_index;
}

uint32_t IRGraph::get_last_submitted_id() const {
    return _lastSubmittedId;
}

std::optional<size_t> IRGraph::determine_batch_size() {
    if (!_metadata.outputs.at(0).shapeFromIRModel.has_value()) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    const ov::PartialShape& firstShape = *_metadata.outputs.at(0).shapeFromIRModel;
    if (firstShape.is_dynamic() || firstShape.rank().get_length() == 0) {
        return std::nullopt;
    }

    const size_t candidateBatchSize = firstShape[utils::BATCH_AXIS].get_max_length();
    if (candidateBatchSize == 0 || candidateBatchSize == utils::DEFAULT_BATCH_SIZE) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    auto checkDescriptorsUseCandidateBatchSize = [candidateBatchSize](const std::vector<IODescriptor>& descriptors) {
        for (const IODescriptor& descriptor : descriptors) {
            OPENVINO_ASSERT(descriptor.shapeFromIRModel.has_value(),
                            "Missing value for the \"shapeFromIRModel\" attribute, I/O descriptor");

            const ov::PartialShape& shapeFromCompiler = descriptor.shapeFromCompiler;
            const ov::PartialShape& shapeFromIRModel = *descriptor.shapeFromIRModel;

            if (shapeFromCompiler.is_dynamic() || shapeFromCompiler.rank().get_length() == 0 ||
                *shapeFromCompiler.begin() != utils::DEFAULT_BATCH_SIZE) {
                return false;
            }

            if (!descriptor.isStateInput && !descriptor.isStateOutput && !descriptor.isShapeTensor) {
                if (shapeFromIRModel.is_dynamic() || shapeFromIRModel.rank().get_length() == 0 ||
                    *shapeFromIRModel.begin() != candidateBatchSize) {
                    return false;
                }
            }
        }

        return true;
    };

    if (!checkDescriptorsUseCandidateBatchSize(_metadata.inputs) ||
        !checkDescriptorsUseCandidateBatchSize(_metadata.outputs)) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    _logger.debug("Batching is handled by the plugin");

    return candidateBatchSize;
}

const std::optional<std::size_t> IRGraph::get_batch_size() const {
    return _batchSize;
}

IRGraph::~IRGraph() {
    // make sure all the context-dependent components are destroyed before the zero context is destroyed
    // if (_handle != nullptr) {
    //     _handle = nullptr;
    // }

    if (!_lastSubmittedEvent.empty()) {
        _lastSubmittedEvent.clear();
    }

    if (_commandQueue != nullptr) {
        _commandQueue.reset();
    }
}

void IRGraph::execute(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                      IRGraph::GraphArguments& args,
                      std::vector<ze_command_list_handle_t>& commandLists,
                      ze_command_queue_handle_t commandQueue,
                      ze_fence_handle_t inferenceFence,
                      ze_event_handle_t event,
                      ze_graph_profiling_pool_handle_t profiling) {
    auto impl = reinterpret_cast<IRGraphImpl*>(_impl.get());

    if (impl == nullptr)
        return;

    impl->executeGraph(zeroInitStruct, args, commandLists, commandQueue, inferenceFence, event, profiling);
}

void IRGraph::getBinding(GraphArguments& args) {
    auto impl = reinterpret_cast<IRGraphImpl*>(_impl.get());

    if (impl == nullptr)
        return;

    impl->getBinding(args);
}

uint64_t IRGraph::get_num_subgraphs() const {
    return _num_of_subgraphs;
}

}  // namespace intel_npu
