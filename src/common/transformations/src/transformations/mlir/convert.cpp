// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/mlir/convert.hpp"

#include <algorithm>
#include <functional>
#include <openvino/op/add.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <unordered_map>

#include "itt.hpp"
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
#include "openvino/core/dimension.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations_visibility.hpp"

using namespace mlir;


static void prepareMLIRKernelWithoutWrapper(mlir::OwningOpRef<mlir::ModuleOp>& module) {
    // A set of default passes that lower any input IR to LLVM
    PassManager pm(module->getContext());

#if 0  // TODO: if TPP is available

    tpp::DefaultPipelineOptions defPipelineOpts{defGpuBackend};
    pm.addPass(tpp::createDefaultPipeline(defPipelineOpts));

#else  // Simplified default lowering to LLVM from LLVM tests

    // Remove empty tensors to avoid converting them into temporary buffers.
    pm.addPass(bufferization::createEmptyTensorEliminationPass());

    pm.addPass(bufferization::createOneShotBufferizePass());
    // TODO: Add deallocation pass/pipeline to avoid memory leaks.

    // Cleanup after bufferization - possibly remove redundant copies.
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());

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

#endif

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
struct MemRef {
    MemRef() = default;

    MemRef(ov::Tensor tensor)
        : allocated(tensor.data()),
          aligned(tensor.data()),
          offset(0),
          shape(tensor.get_shape().begin(), tensor.get_shape().end()) {
        strides.resize(tensor.get_shape().size());
        const auto& byte_strides = tensor.get_strides();
        auto element_size = tensor.get_element_type().size();
        for (size_t i = 0; i < strides.size(); ++i) {
            assert(byte_strides[i] % element_size == 0);
            // TODO: handle case when stride is not aligned (restrict at OV API level)
            strides[i] = byte_strides[i] / element_size;
            //std::cout << "stride [" << i << "] = " << strides[i] << "\n";
        }
    }

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

class MLIREvaluate {
    OwningOpRef<mlir::ModuleOp> module;  // FIXME: needs to be kept?
    std::unique_ptr<ExecutionEngine> engine;

public:
    MLIREvaluate(OwningOpRef<mlir::ModuleOp> _module) : module(std::move(_module)) {
        if (true) {
            std::cout << "[ DEBUG ] Source MLIR:\n";
            std::cerr << "-----------------------------------------\n";
            module->dump();
            std::cout << "-----------------------------------------\n";
        }

        prepareMLIRKernelWithoutWrapper(module);

        if (true) {
            std::cerr << "[ DEBUG ] Target LLVM:\n";
            std::cerr << "-----------------------------------------\n";
            module->dump();
            std::cerr << "-----------------------------------------\n";
        }

        auto optPipeline = mlir::makeOptimizingTransformer(2,
                                                           /*sizeLevel=*/0,  // FIXME: HARDCODED
                                                           /*targetMachine=*/nullptr);

        mlir::ExecutionEngineOptions engineOptions;
        engineOptions.transformer = optPipeline;  // opt level looks to be overriden in lowerToLLVMIR, but is still used
                                                  // in `create` independently
        engineOptions.llvmModuleBuilder = lowerToLLVMIR;
        auto maybeEngine = mlir::ExecutionEngine::create(module.get(), engineOptions);
        assert(maybeEngine && "failed to construct an execution engine");
        engine = std::move(maybeEngine.get());
    }

    bool invoke_packed(std::vector<void*>& args) {
        auto invocationResult = engine->invokePacked("entry", args);
        if (invocationResult) {
            llvm::errs() << "JIT invocation failed\n";
            return false;
        }
        return true;
    }
};

typedef std::vector<std::tuple<ov::element::Type, ov::PartialShape>> OVOutputTypes;

class OPENVINO_API MLIROp : public ov::op::Op {
    std::shared_ptr<MLIREvaluate> engine;
    OVOutputTypes output_types;

public:
    OPENVINO_OP("MLIROp");

    MLIROp(const ov::OutputVector& args, std::shared_ptr<MLIREvaluate> engine, const OVOutputTypes& output_types)
        : Op(args),
          engine(engine),
          output_types(output_types) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(output_types.size());
        for (size_t i = 0; i < output_types.size(); ++i) {
            set_output_type(i, std::get<0>(output_types[i]), std::get<1>(output_types[i]));
        }
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<MLIROp>(new_args, engine, output_types);
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        // FIXME: Assumes an output contains only one tensor which shape is just propagated from inputs[0].
        // TODO: Involve internal for a partition shape propagation or rely on symbolic shape info.
        outputs[0].set_shape(inputs[0].get_shape());

        std::vector<MemRef> memref_args;
        for (size_t i = 0; i < inputs.size(); ++i) {
            memref_args.push_back(MemRef(inputs[i]));
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            memref_args.push_back(MemRef(outputs[i]));
        }
        std::vector<void*> args;

        std::for_each(memref_args.begin(), memref_args.end(), [&args](MemRef& x) {
            x.append_to_packed_args(args);
        });

        std::cerr << "[ INFO ] Running kernel in MLIROp::evaluate\n";
        return engine->invoke_packed(args);
    }

    bool has_evaluate() const override {
        return true;
    }
};

mlir::Location createLayerLocation(mlir::MLIRContext* ctx, const std::string& layerName, const std::string& layerType) {
    const auto layerNameAttr = mlir::StringAttr::get(ctx, layerName);
    const auto nameLoc = mlir::NameLoc::get(layerNameAttr);

    SmallVector<mlir::NamedAttribute> fields;
    fields.emplace_back(mlir::StringAttr::get(ctx, "type"), mlir::StringAttr::get(ctx, layerType));
    fields.emplace_back(mlir::StringAttr::get(ctx, "name"), layerNameAttr);
    auto metadata = mlir::DictionaryAttr::get(ctx, fields);

    return mlir::FusedLoc::get(ctx, {nameLoc}, metadata);
}

SmallVector<int64_t> importShape(const ov::PartialShape& shape) {
    SmallVector<int64_t> out(shape.rank().get_length());
    // TODO: Add support for dynamically ranked shapes
    for (size_t i = 0; i < out.size(); ++i) {
        const auto& dim = shape[i];
        out[i] = dim.is_static() ? dim.get_length() : mlir::ShapedType::kDynamic;
    }
    return out;
}

mlir::IntegerType getInt1Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 1);
}

mlir::IntegerType getInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4);
}

mlir::IntegerType getInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8);
}

mlir::IntegerType getInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16);
}

mlir::IntegerType getInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32);
}

mlir::IntegerType getInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64);
}

mlir::IntegerType getSInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Signed);
}

mlir::IntegerType getSInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
}

mlir::IntegerType getSInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Signed);
}

mlir::IntegerType getSInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
}

mlir::IntegerType getSInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
}

mlir::IntegerType getUInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Unsigned);
}

mlir::IntegerType getUInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
}

mlir::IntegerType getUInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);
}

mlir::IntegerType getUInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
}

mlir::IntegerType getUInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
}

mlir::IntegerType getBool8Type(mlir::MLIRContext* ctx) {
    // Signless 8-bit integer use for BOOL, to distinguish it from U8
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signless);
}

mlir::Type importPrecision(mlir::MLIRContext* ctx, const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::Type_t::f64:
        return mlir::Float64Type::get(ctx);
    case ov::element::Type_t::f32:
        return mlir::Float32Type::get(ctx);
    case ov::element::Type_t::f16:
        return mlir::Float16Type::get(ctx);
    case ov::element::Type_t::bf16:
        return mlir::BFloat16Type::get(ctx);
    case ov::element::Type_t::i64:
        return getSInt64Type(ctx);
    case ov::element::Type_t::u64:
        return getUInt64Type(ctx);
    case ov::element::Type_t::i32:
        return getSInt32Type(ctx);
    case ov::element::Type_t::u32:
        return getUInt32Type(ctx);
    case ov::element::Type_t::i16:
        return getSInt16Type(ctx);
    case ov::element::Type_t::u16:
        return getUInt16Type(ctx);
    case ov::element::Type_t::i8:
        return getSInt8Type(ctx);
    case ov::element::Type_t::u8:
        return getUInt8Type(ctx);
    case ov::element::Type_t::i4:
        return getSInt4Type(ctx);
    case ov::element::Type_t::u4:
        return getUInt4Type(ctx);
    case ov::element::Type_t::boolean:
        return getBool8Type(ctx);
    default:
        OPENVINO_THROW("Unsupported element_type: ", precision);
    }
}

mlir::RankedTensorType importTensor(mlir::MLIRContext* ctx,
                                    const ov::PartialShape& shape,
                                    const ov::element::Type& elemType) {
    return mlir::RankedTensorType::get(ArrayRef(importShape(shape)), importPrecision(ctx, elemType));
}

mlir::Location createLocation(mlir::MLIRContext* ctx, std::shared_ptr<ov::Node> node) {
    return createLayerLocation(ctx, node->get_friendly_name(), node->get_type_name());
}

namespace std {

template <typename T>
size_t getHash(const T& val) {
    return std::hash<T>()(val);
}

template <typename T, typename... Args>
size_t getHash(const T& val, Args&&... args) {
    return llvm::hash_combine(getHash(val), getHash(std::forward<Args>(args)...));
}

template <>
struct hash<ov::Output<ov::Node>> final {
    size_t operator()(const ov::Output<ov::Node>& out) const {
        return getHash(out.get_node(), out.get_index());
    }
};
}  // namespace std


MemRefType convertTensorToMemRef(TensorType tensorType) {
    ArrayRef<int64_t> shape = tensorType.getShape();
    Type elementType = tensorType.getElementType();
    return MemRefType::get(shape, elementType);
}


SmallVector<mlir::Type> tensorsToMemRefs(SmallVector<mlir::Type> tensors) {
    SmallVector<mlir::Type> out;
    out.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        out.push_back(convertTensorToMemRef(dyn_cast<TensorType>(tensor)));
    }
    return out;
}


SmallVector<mlir::Type> get_types_for_values(mlir::MLIRContext* context, const ov::OutputVector& values) {
    SmallVector<mlir::Type> types;
    types.reserve(values.size());
    for (const auto& output : values) {
        types.push_back(importTensor(context, output.get_partial_shape(), output.get_element_type()));
    }
    return types;
}

class ConversionContext {
public:
    using Convertor = std::function<void(ConversionContext&, std::shared_ptr<ov::Node>)>;
    using NodeOutputMap = std::unordered_map<ov::Output<ov::Node>, mlir::Value>;

    static const std::map<ov::DiscreteTypeInfo, Convertor> convertors;
    mlir::MLIRContext* context;
    mlir::OpBuilder* block_builder;
    NodeOutputMap nodeOutputMap;

    ConversionContext(mlir::MLIRContext* context, mlir::OpBuilder* block_builder)
        : context(context),
          block_builder(block_builder) {}

    SmallVector<mlir::Value> getInputs(std::shared_ptr<ov::Node> node) {
        SmallVector<mlir::Value> out;
        out.reserve(node->get_input_size());
        for (const auto& input : node->inputs()) {
            out.push_back(nodeOutputMap.at(input.get_source_output()));
        }
        return out;
    }

    void addOutputs(std::shared_ptr<ov::Node> node, mlir::Operation* op) {
        const auto results = op->getOpResults();

        OPENVINO_ASSERT(
            results.size() == node->get_output_size(),
            "Mismatch between original Node '{0}' number of outputs '{1}' and created number of outputs '{2}'",
            node->get_friendly_name(),
            node->get_output_size(),
            results.size());

        for (const auto& res : results) {
            nodeOutputMap.emplace(node->output(res.getResultNumber()), res);
        }
    }

    mlir::OpBuilder& builder() {
        return *block_builder;
    }

    void convert(std::shared_ptr<ov::Node> node) {
        convertors.at(node->get_type_info())(*this, node);
    }
};

template <typename TargetOp>
struct ConvertBinary {
    void operator()(ConversionContext& context, std::shared_ptr<ov::Node> node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        // TODO: Support broadcasts
        const auto inputs = context.getInputs(node);
        auto outType = cast<mlir::ShapedType>(inputs[0].getType());
        // Named binary ops directly overwrite data in `outs` buffer so, there is no need to provide non-empty
        // destination at the tensor-level.
        // Use `tensor.empty` to avoid temporary buffer allocation and memcpy after bufferization.
        llvm::SmallVector<Value> dynamicSizes;
        for (auto [idx, dim] : llvm::enumerate(outType.getShape())) {
            if (!mlir::ShapedType::isDynamic(dim))
                continue;
            auto dimSize = builder.create<tensor::DimOp>(loc, inputs[0], idx);
            dynamicSizes.push_back(dimSize);
        }
        auto empty = builder.create<tensor::EmptyOp>(loc, outType, dynamicSizes);
        auto op = builder.create<TargetOp>(loc, mlir::ValueRange{inputs[0], inputs[1]}, mlir::ValueRange{empty});
        context.addOutputs(node, op);
    }
};

const std::map<ov::DiscreteTypeInfo, ConversionContext::Convertor> ConversionContext::convertors = {
    {ov::op::v1::Add::get_type_info_static(), Convertor(ConvertBinary<linalg::AddOp>())}};

mlir::OwningOpRef<mlir::ModuleOp> ngraph_to_mlir(MLIRContext* context,
                                                 const ov::OutputVector& inputs,
                                                 const ov::NodeVector& nodes,
                                                 const ov::OutputVector& outputs) {
    auto inputTypes = tensorsToMemRefs(get_types_for_values(context, inputs));
    auto outputTypes = tensorsToMemRefs(get_types_for_values(context, outputs));

    const auto moduleLoc = createLayerLocation(context, "module", "Module");
    auto module = mlir::ModuleOp::create(moduleLoc, StringRef("fragment_name"));

    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody() /* TODO: building log here */);

    const auto funcLoc = createLayerLocation(context, "entry", "Func");
    auto memref_args = inputTypes;
    memref_args.append(outputTypes);
    const auto funcType = mlir::FunctionType::get(context, ArrayRef(memref_args), ArrayRef(SmallVector<mlir::Type>()));
    auto func = moduleBuilder.create<mlir::func::FuncOp>(funcLoc, "entry", funcType);
    auto block_builder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock() /* TODO: Add logger here */);

    ConversionContext conversion_context(context, &block_builder);

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto funcInputVal = func.getArgument(i);
        // transition from memref enclosure to tensor interior
        auto loc = createLocation(context, inputs[i].get_node_shared_ptr());
        auto tensor = block_builder.create<bufferization::ToTensorOp>(loc, funcInputVal, /*restrict = */ true);
        conversion_context.nodeOutputMap.emplace(inputs[i], tensor);
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto node = nodes[i];
        conversion_context.convert(node);
    }

    SmallVector<mlir::Value> funcOutputs;
    funcOutputs.reserve(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = conversion_context.nodeOutputMap.at(outputs[i]);
        auto memref = func.getArgument(i + inputs.size());
        auto loc = createLocation(context, outputs[i].get_node_shared_ptr());
        // Ensure the result is stored in the provided function argument.
        // Mark as restrict to avoid temporary buffer and copy.
        // Mark as writable to ensure the output can be written to the buffer.
        block_builder.create<bufferization::MaterializeInDestinationOp>(loc,
                                                                        TypeRange{},
                                                                        tensor,
                                                                        memref,
                                                                        /*restrict=*/true,
                                                                        /*writable=*/true);
    }

    const auto retLoc = createLayerLocation(context, "output", "Output");
    block_builder.create<mlir::func::ReturnOp>(retLoc, ArrayRef(SmallVector<Value>()));

    return module;
}

class AddLowering : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("AddLowering", "0");
    explicit AddLowering(mlir::MLIRContext* context) {
        auto pattern = ov::pass::pattern::wrap_type<ov::op::v1::Add>(
            {ov::pass::pattern::any_input(), ov::pass::pattern::any_input()});

        auto callback = [=, context](ov::pass::pattern::Matcher& m) {
            std::cout << "[ INFO ] Matched AddLowering\n";
            auto add = m.get_match_root();

            mlir::OwningOpRef<mlir::ModuleOp> module;

            // FIXME: Suppose no broadcast
            module = ngraph_to_mlir(context, add->input_values(), {add}, add->outputs());

            auto expected_outputs = add->outputs();
            OVOutputTypes output_types;
            for (size_t i = 0; i < expected_outputs.size(); ++i) {
                output_types.push_back(
                    std::make_tuple(expected_outputs[i].get_element_type(), expected_outputs[i].get_partial_shape()));
            }
            auto replacement = std::make_shared<MLIROp>(add->input_values(),
                                                        std::make_shared<MLIREvaluate>(std::move(module)),
                                                        output_types);

            replace_node(add, replacement);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "AddLowering");
        register_matcher(m, callback);
    }
};


void injectMLIR(std::shared_ptr<ov::Model> model, MLIRContext* context) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(true);
    manager.register_pass<AddLowering>(context);
    manager.run_passes(model);
}


MLIRContext* get_shared_mlir_context() {
    // Gives MLIRContext instance shared for entire OV process and initialized once upon the initial request
    // FIXME: Bind with OpenVINO lifetime in the sutable class instead of dirty tricking with static lifetime

    static std::shared_ptr<MLIRContext> context;

    if (!context) {

        // Initialize the LLVM machinery
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        // Initialize GPU-related LLVM machinery
        // tpp::initializeGpuTargets();

        // Add the following to include *all* MLIR Core dialects, or selectively
        // include what you need like above. You only need to register dialects that
        // will be *parsed* by the tool, not the one generated
        DialectRegistry registry;
        // registry.insert<mlir::xsmm::XsmmDialect>();
        // registry.insert<mlir::check::CheckDialect>();
        // registry.insert<mlir::perf::PerfDialect>();

        registerAllDialects(registry);
        registerAllExtensions(registry);
        registerAllToLLVMIRTranslations(registry);
        mlir::linalg::registerTransformDialectExtension(registry);
        mlir::tensor::registerTransformDialectExtension(registry);

        context = std::make_shared<MLIRContext>(registry);

        context->loadDialect<mlir::func::FuncDialect>();
        context->loadDialect<mlir::linalg::LinalgDialect>();
        context->loadDialect<mlir::bufferization::BufferizationDialect>();
    }

    return context.get();
}

void ov::pass::transformMLIR(std::shared_ptr<ov::Model> model) {
    injectMLIR(model, get_shared_mlir_context());
}
