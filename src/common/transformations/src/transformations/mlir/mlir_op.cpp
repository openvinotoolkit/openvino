// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_op.hpp"

#include <vector>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

// TODO: Prune unused headers -- it's hard to understand needed ones
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#ifdef TPP_MLIR // If TPP is available
#include "TPP/PassBundles.h"
#include "TPP/Passes.h"
#endif

#ifdef GRAPH_COMPILER
#include "gc/Transforms/Passes.h"

#ifdef GC_USE_IMEX // GC_GPU requires IMEX support
#include "gc/Utils/Error.h"
#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#endif
#endif

namespace {

using namespace mlir;

using NodePtr = std::shared_ptr<ov::Node>;
using SymbolPtr = std::shared_ptr<ov::Symbol>;

void prepareMLIRKernelWithoutWrapper(mlir::OwningOpRef<mlir::ModuleOp>& module, ov::mlir::MlirMode mode) {
    PassManager pm(module->getContext());

    switch (mode) {
#ifdef TPP_MLIR
        case ov::mlir::MLIR_MODE_TPP: {
            tpp::DefaultPipelineOptions defPipelineOpts;
            pm.addPass(tpp::createDefaultPipeline(defPipelineOpts));
            break;
        }
#endif
#ifdef GRAPH_COMPILER
        case ov::mlir::MLIR_MODE_GC: {
            gc::populateCPUPipeline(pm);
            break;
        }
#endif
        default: {
            assert(ov::mlir::MLIR_MODE_DEFAULT);
            // Cleanup before bufferization.
            // Simplifies IR to allow better bufferization.
            pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
            pm.addNestedPass<func::FuncOp>(createCSEPass());

            // Remove empty tensors to avoid converting them into temporary buffers.
            pm.addPass(bufferization::createEmptyTensorEliminationPass());

            pm.addPass(bufferization::createOneShotBufferizePass());
            pm.addNestedPass<func::FuncOp>(bufferization::createFinalizingBufferizePass());

            // Cleanup after bufferization - possibly remove redundant copies.
            pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
            pm.addNestedPass<func::FuncOp>(createCSEPass());

            // Deallocation pipeline to avoid memory leaks from created temporary buffers.
            pm.addPass(memref::createExpandReallocPass(/*emitDeallocs=*/false));
            pm.addPass(createCanonicalizerPass());
            bufferization::DeallocationOptions deallocOpts;
            deallocOpts.privateFuncDynamicOwnership = false;
            pm.addPass(bufferization::createOwnershipBasedBufferDeallocationPass(deallocOpts));
            pm.addPass(createCanonicalizerPass());
            pm.addPass(bufferization::createBufferDeallocationSimplificationPass());
            pm.addPass(bufferization::createLowerDeallocationsPass());
            pm.addPass(createCSEPass());
            pm.addPass(createCanonicalizerPass());

            // Blanket-convert any remaining high-level vector ops to loops if any remain.
            pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
            // pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
            //  Blanket-convert any remaining linalg ops to loops if any remain.
            pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
            // Blanket-convert any remaining affine ops if any remain.
            pm.addPass(createLowerAffinePass());
            // Convert SCF to CF (always needed).
            pm.addPass(createConvertSCFToCFPass());
            // Sprinkle some cleanups.
            pm.addPass(createCanonicalizerPass());
            pm.addPass(createCSEPass());
            // Blanket-convert any remaining linalg ops to LLVM if any remain.
            // pm.addPass(createConvertLinalgToLLVMPass());  // no such pass
            // Convert vector to LLVM (always needed).
            pm.addPass(createConvertVectorToLLVMPass());
            // Convert Math to LLVM (always needed).
            pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
            // Expand complicated MemRef operations before lowering them.
            pm.addPass(memref::createExpandStridedMetadataPass());
            // The expansion may create affine expressions. Get rid of them.
            pm.addPass(createLowerAffinePass());
            // Convert MemRef to LLVM (always needed).
            // pm.addPass(memref::createExpandOpsPass());
            pm.addPass(createFinalizeMemRefToLLVMConversionPass());
            // Convert Func to LLVM (always needed).
            pm.addPass(createConvertFuncToLLVMPass());
            // Convert Index to LLVM (always needed).
            pm.addPass(createConvertIndexToLLVMPass());
            // Convert remaining unrealized_casts (always needed).
            pm.addPass(createReconcileUnrealizedCastsPass());
        }
    }

    auto result = pm.run(module.get());
    if (failed(result)) {
        llvm::errs() << "ERROR: Failed to lower IR to LLVM dialect\n";
        module->print(llvm::errs());
    }
}

std::unique_ptr<llvm::Module> lowerToLLVMIR(Operation* module, llvm::LLVMContext& llvmContext) {
    // Default lowering for mlir-cpu-runner
    auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
    assert(llvmModule);

    // Target machine, null if not specified
    std::unique_ptr<llvm::TargetMachine> targetMachine;

    std::string triple = "x86_64-linux-gnu";
    std::string cpuName = "alderlake";  // sapphirerapids, nehalem, etc.
    std::string fpuName = "avx2";       //  sse4.2, avx, avx2, avx512bf16, etc.
    bool printLLVM = false;
    auto codeGenOpt = 2;

    // Specify target machine
    if (!triple.empty() && !cpuName.empty()) {
        std::string error;
        const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
        if (!target) {
            llvm::errs() << "Error while looking up target triple: ";
            llvm::errs() << error << "\n";
            return nullptr;
        }

        // These options should force fused MLA, but they don't. :/
        // Adding unsafe math attribute to functions below do the trick.
        llvm::TargetOptions targetOptions;
        targetOptions.UnsafeFPMath = true;
        targetOptions.AllowFPOpFusion = llvm::FPOpFusion::FPOpFusionMode::Fast;
        targetMachine.reset(target->createTargetMachine(triple,
                                                        cpuName,
                                                        "+" + fpuName,
                                                        targetOptions,
                                                        /* reloc model */ std::nullopt,
                                                        /* code model */ std::nullopt,
                                                        llvm::CodeGenOptLevel(codeGenOpt)));
        if (!targetMachine) {
            llvm::errs() << "Error while looking up target CPU: ";
            llvm::errs() << cpuName << "\n";
            return nullptr;
        }
    }

    // Run the optimized pipeline
    int sizeLevel = 0;
    auto optPipeline = makeOptimizingTransformer(codeGenOpt, sizeLevel, targetMachine.get());
    if (auto err = optPipeline(llvmModule.get())) {
        llvmModule->print(llvm::errs(), nullptr);
        llvm::errs() << "Error while passing through the LLVM pipeline: ";
        llvm::errs() << err << "\n";
        return nullptr;
    }

    // MLIR doesn't lower LLVM with fast-math flags, but we need that, so we
    // add for each function, to get FMAs and other goodies.
    for (auto& func : llvmModule->functions()) {
        func.addFnAttr("unsafe-fp-math", "true");
    }

    if (printLLVM)
        llvmModule->print(llvm::outs(), nullptr);

    return llvmModule;
}

// TODO: u4/i4 types are not supported
struct MemRefDescriptor {
    MemRefDescriptor() = default;

    MemRefDescriptor    (ov::Tensor tensor, const ov::Shape& module_input_shape)
        : allocated(tensor.data()),
          aligned(tensor.data()),
          offset(0),
          shape(module_input_shape.begin(), module_input_shape.end()) {
        if (shape.size() != tensor.get_shape().size()) {
            // validate that the shape difference is due to trailing '1's
            for (size_t i = 0; i < shape.size(); ++i) {
                if (shape[i] != tensor.get_shape()[i]) {
                    OPENVINO_THROW("Mismatch in shape sizes");
                }
            }
            for (size_t i = shape.size(); i < tensor.get_shape().size(); ++i) {
                if (tensor.get_shape()[i] != 1) {
                    OPENVINO_THROW("Mismatch in shape sizes");
                }
            }
        }
        strides.resize(shape.size());
        const auto& byte_strides = tensor.get_strides();
        auto element_size = tensor.get_element_type().size();
        for (size_t i = 0; i < strides.size(); ++i) {
            assert(byte_strides[i] % element_size == 0);
            // TODO: handle case when stride is not aligned (restrict at OV API level)
            strides[i] = byte_strides[i] / element_size;
            //std::cerr << "stride [" << i << "] = " << strides[i] << "\n";
        }
    }

    MemRefDescriptor    (ov::Tensor tensor)
        : MemRefDescriptor(tensor, tensor.get_shape()) {}

    void* allocated;
    void* aligned;
    int64_t offset;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

    void append_to_packed_args(std::vector<void*>& args) {
        args.push_back(&allocated);
        args.push_back(&aligned);
        args.push_back(&offset);
        for (size_t i = 0; i < shape.size(); ++i) {
            args.push_back(&shape[i]);
        }
        for (size_t i = 0; i < strides.size(); ++i) {
            args.push_back(&strides[i]);
        }
    }
};

} // namespace

namespace ov {
namespace mlir {

using namespace ::mlir;

std::shared_ptr<MLIREvaluateBase> MLIREvaluateBase::create(OwningOpRef<ModuleOp> module,
                                                           MlirMode mode,
                                                           std::shared_ptr<ov::EvaluationContext> loweringContext) {
    switch (mode) {
        #ifdef GC_USE_IMEX // GC_GPU requires IMEX support
        case MLIR_MODE_GC_GPU:
            return std::make_shared<MLIREvaluateGcGPU>(std::move(module), loweringContext);
        #endif
        case MLIR_MODE_TPP:
        case MLIR_MODE_GC:
        case MLIR_MODE_DEFAULT:
            return std::make_shared<MLIREvaluate>(std::move(module), mode);
        default:
            OPENVINO_THROW("Unsupported MLIR mode");
    }
}

#ifdef GC_USE_IMEX // GC_GPU requires IMEX support

cl_device_id extract_device_from_context(cl_context context) {
    size_t devices_size;
    cl_int err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &devices_size);
    if (err != CL_SUCCESS) {
        OPENVINO_THROW("Error getting context info: ", err);
    }
    if (devices_size / sizeof(cl_device_id) != 1) {
        OPENVINO_THROW("Expected exactly one device in the context, got ", devices_size);
    }

    cl_device_id devices;
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size, &devices, NULL);
    if (err != CL_SUCCESS) {
        OPENVINO_THROW("Error getting device IDs: ", err);
    }

    return devices;
}

MLIREvaluateGcGPU::MLIREvaluateGcGPU(OwningOpRef<mlir::ModuleOp> _module, std::shared_ptr<ov::EvaluationContext> loweringContext) {
    OPENVINO_MLIR_DEBUG_PRINT(
        "[ DEBUG ] Source MLIR:\n"
        "-----------------------------------------\n");
    OPENVINO_MLIR_DEBUG(_module->dump());
    OPENVINO_MLIR_DEBUG_PRINT(
        "-----------------------------------------\n");

    gc::gpu::OclModuleBuilderOpts opts;
    OPENVINO_MLIR_DEBUG(opts.printIr = true);
    gc::gpu::OclModuleBuilder builder(std::move(_module), opts);

    auto it = loweringContext->find(ov::intel_gpu::ocl_context.name());
    if (it == loweringContext->end()) {
        OPENVINO_THROW("No cl_context provided for OpenCL execution");
    }
    auto context = reinterpret_cast<cl_context>(it->second.as<ov::intel_gpu::gpu_handle_param>());
    // assuming there's always one device per context
    auto device = extract_device_from_context(context);

    OPENVINO_MLIR_DEBUG_PRINT(
        "[ DEBUG ] Target LLVM:\n"
        "-----------------------------------------\n");
    if (auto mod = builder.build(device, context)) {
        module = *mod;
    } else {
        OPENVINO_THROW("Failed to build gc::gpuOclModule module");
    }
    OPENVINO_MLIR_DEBUG_PRINT(
        "-----------------------------------------\n");
};

bool MLIREvaluateGcGPU::invoke(const ov::TensorVector& inputs, ov::TensorVector& outputs, const ov::EvaluationContext& evaluationContext) {
    gc::gpu::OclContext ctx = build_ocl_context(evaluationContext);
    gc::gpu::StaticExecutor exec(module);

    auto it = evaluationContext.find(ov::internal::mlir_meta::is_kernel_arg_usm.name());
    if (it == evaluationContext.end()) {
        OPENVINO_THROW("No is_kernel_arg_usm provided for OpenCL execution");
    }
    std::vector<bool> arg_types = it->second.as<std::vector<bool>>();

    for (size_t i = 0; i < inputs.size(); ++i) {
        exec.arg(inputs[i].data(), arg_types[i]);
    }
    for (size_t i = 0, j = inputs.size(); i < outputs.size(); ++i, ++j) {
        exec.arg(outputs[i].data(), arg_types[j]);
    }
    exec(ctx);
    maybe_set_result_event(evaluationContext, ctx);
    return true;
}

bool MLIREvaluateGcGPU::invoke_packed(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) {
    gc::gpu::OclContext ctx = build_ocl_context(evaluationContext);
    gc::gpu::DynamicExecutor exec(module);

    auto it = evaluationContext.find(ov::internal::mlir_meta::is_kernel_arg_usm.name());
    if (it == evaluationContext.end()) {
        OPENVINO_THROW("No is_kernel_arg_usm provided for OpenCL execution");
    }
    std::vector<bool> argTypes = it->second.as<std::vector<bool>>();
    for (size_t i = 0; i < args.size(); i+=4) {
        exec.arg(
            /*alignedPtr=*/args[i],
            /*rank=*/reinterpret_cast<size_t>(args[i + 1]),
            /*shape=*/reinterpret_cast<int64_t*>(args[i + 2]),
            /*strides=*/reinterpret_cast<int64_t*>(args[i + 3]),
            /*isUsm=*/argTypes[i]
        );
    }
    exec(ctx);
    maybe_set_result_event(evaluationContext, ctx);
    return true;
}

void MLIREvaluateGcGPU::maybe_set_result_event(const ov::EvaluationContext& evaluationContext, gc::gpu::OclContext& ctx) {
    // case with in-order queue where we don't need to return an event
    if (ctx.lastEvent == nullptr)
        return;
    auto it = evaluationContext.find(ov::internal::mlir_meta::result_event.name());
    if (it == evaluationContext.end()) {
        OPENVINO_THROW("No result_event provided for OpenCL execution");
    }
    cl_event* ev = reinterpret_cast<cl_event*>(it->second.as<void**>());
    *ev = ctx.lastEvent;
}

gc::gpu::OclContext MLIREvaluateGcGPU::build_ocl_context(const ov::EvaluationContext& evaluationContext) {
    auto it = evaluationContext.find(ov::intel_gpu::ocl_queue.name());
    if (it == evaluationContext.end()) {
        OPENVINO_THROW("No queue provided for OpenCL execution");
    }
    cl_command_queue queue = reinterpret_cast<cl_command_queue>(it->second.as<void*>());

    uint32_t waitListLen = 0;
    std::vector<void*> waitList;
    bool foundWaitList = false;

    it = evaluationContext.find(ov::internal::mlir_meta::wait_list.name());
    if (it != evaluationContext.end()) {
        waitList = it->second.as<std::vector<void*>>();
        waitListLen = waitList.size();
        foundWaitList = true;
    }

    return gc::gpu::OclContext(module->runtime, queue, /*createEvents=*/foundWaitList,
                               waitListLen, reinterpret_cast<cl_event*>(waitList.data()));
}

#endif // GC_USE_IMEX

MLIREvaluate::MLIREvaluate(OwningOpRef<mlir::ModuleOp> _module, MlirMode mode) :
    module(std::move(_module)) {

    OPENVINO_MLIR_DEBUG_PRINT(
        "[ DEBUG ] Source MLIR:\n"
        "-----------------------------------------\n");
    OPENVINO_MLIR_DEBUG(module->dump());
    OPENVINO_MLIR_DEBUG_PRINT(
        "-----------------------------------------\n");

    prepareMLIRKernelWithoutWrapper(module, mode);

    OPENVINO_MLIR_DEBUG_PRINT(
        "[ DEBUG ] Target LLVM:\n"
        "-----------------------------------------\n");
    OPENVINO_MLIR_DEBUG(module->dump());
    OPENVINO_MLIR_DEBUG_PRINT(
        "-----------------------------------------\n");

    auto optPipeline = mlir::makeOptimizingTransformer(2,
                                                        /*sizeLevel=*/0,  // FIXME: HARDCODED
                                                        /*targetMachine=*/nullptr);

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;  // opt level looks to be overriden in lowerToLLVMIR, but is still used
                                                // in `create` independently
    engineOptions.llvmModuleBuilder = lowerToLLVMIR;
    auto maybeEngine = mlir::ExecutionEngine::create(module.get(), engineOptions);
    if (maybeEngine) {
        engine = std::move(maybeEngine.get());
    } else {
        llvm::errs() << "failed to construct an execution engine\n";
        abort();
    }
}

bool MLIREvaluate::invoke_packed(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) {
    auto invocationResult = engine->invokePacked("entry", args);
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return false;
    }
    return true;
}

MLIROp::MLIROp(const ov::OutputVector& args, std::shared_ptr<MLIREvaluateBase> engine, const OVOutputTypes& output_types, const DimensionsMap& dimensions_map)
    : Op(args),
        engine(engine),
        output_types(output_types),
        dimensions_map(dimensions_map) {
    constructor_validate_and_infer_types();
}

void MLIROp::validate_and_infer_types() {
    set_output_size(output_types.size());
    for (size_t i = 0; i < output_types.size(); ++i) {
        set_output_type(i, std::get<0>(output_types[i]), std::get<1>(output_types[i]));
    }
}

NodePtr MLIROp::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<MLIROp>(new_args, engine, output_types, dimensions_map);
}

bool MLIROp::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs, const ov::EvaluationContext& evaluationContext) const {
    if (!engine->requires_packed_args()) {
        return engine->invoke(inputs, outputs, evaluationContext);
    }

    std::vector<MemRefDescriptor> memref_args;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& initial_shape = get_input_shape(i);
        memref_args.push_back(MemRefDescriptor(inputs[i], initial_shape));
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        // TODO: Optimize by adding all dimensions to dimensions_map, not only dynamic
        Shape target;
        PartialShape expected = get_output_partial_shape(i);
        for(size_t j = 0; j < expected.size(); ++j) {
            auto dim = expected[j];
            if(dim.is_dynamic()) {
                int input_index, dim_index;
                std::tie(input_index, dim_index) = dimensions_map[i][j];
                target.push_back(inputs[input_index].get_shape()[dim_index]);
            } else {
                target.push_back(dim.get_length());
            }
        }
        //std::cerr << "[ DEBUG ] Set outputs[" << i << "].shape(" << target << ")\n";
        outputs[i].set_shape(target);
        memref_args.push_back(MemRefDescriptor(outputs[i]));
    }
    std::vector<void*> args;

    std::for_each(memref_args.begin(), memref_args.end(), [&args](MemRefDescriptor& x) {
        x.append_to_packed_args(args);
    });

    //std::cerr << "[ INFO ] Running kernel in MLIROp::evaluate\n";
    return engine->invoke_packed(args, evaluationContext);
}

bool MLIROp::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    return evaluate(outputs, inputs, ov::EvaluationContext());
}

bool MLIROp::has_evaluate() const {
    return true;
}

} // namespace mlir
} // namespace ov