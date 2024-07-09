// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/mlir/convert.hpp"

#include <algorithm>
#include <functional>
#include <openvino/op/add.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/divide.hpp>

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
#include "openvino/core/symbol.hpp"

#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace {

using namespace mlir;

using NodePtr = std::shared_ptr<ov::Node>;
using SymbolPtr = std::shared_ptr<ov::Symbol>;



void prepareMLIRKernelWithoutWrapper(mlir::OwningOpRef<mlir::ModuleOp>& module) {
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
        if (maybeEngine) {
            engine = std::move(maybeEngine.get());
        } else {
            llvm::errs() << "failed to construct an execution engine\n";
            abort();
        }
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

class MLIROp : public ov::op::Op {
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

    NodePtr clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<MLIROp>(new_args, engine, output_types);
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
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

mlir::Location createLocation(mlir::MLIRContext* ctx, NodePtr node) {
    return createLayerLocation(ctx, node->get_friendly_name(), node->get_type_name());
}

} // namespace


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


namespace {

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
    using Convertor = std::function<void(ConversionContext&, NodePtr)>;
    using NodeOutputMap = std::unordered_map<ov::Output<ov::Node>, mlir::Value>;

    static const std::map<ov::DiscreteTypeInfo, Convertor> convertors;
    mlir::MLIRContext* context;
    mlir::OpBuilder* block_builder;
    NodeOutputMap nodeOutputMap;

    ConversionContext(mlir::MLIRContext* context, mlir::OpBuilder* block_builder)
        : context(context),
          block_builder(block_builder) {}

    SmallVector<mlir::Value> getInputs(NodePtr node) {
        SmallVector<mlir::Value> out;
        out.reserve(node->get_input_size());
        for (const auto& input : node->inputs()) {
            out.push_back(nodeOutputMap.at(input.get_source_output()));
        }
        return out;
    }

    void addOutputs(NodePtr node, mlir::Operation* op) {
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

    void convert(NodePtr node) {
        convertors.at(node->get_type_info())(*this, node);
    }
};

template <typename TargetOp>
struct ConvertBinary {
    void operator()(ConversionContext& context, NodePtr node) {
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
    {ov::op::v1::Add::get_type_info_static(), Convertor(ConvertBinary<linalg::AddOp>())},
    {ov::op::v1::Subtract::get_type_info_static(), Convertor(ConvertBinary<linalg::SubOp>())},
    {ov::op::v1::Multiply::get_type_info_static(), Convertor(ConvertBinary<linalg::MulOp>())},
    {ov::op::v1::Divide::get_type_info_static(), Convertor(ConvertBinary<linalg::DivOp>())},
};

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


using InputVector = std::vector<ov::Input<ov::Node>>;


struct Subgraph {
    ov::NodeVector nodes;
    ov::OutputVector inputs;
    ov::OutputVector outputs;
    std::vector<InputVector> output_consumers;

    // Consumes other subgraph
    void merge (Subgraph& other) {
        nodes.insert(nodes.end(), other.nodes.begin(), other.nodes.end());
    }
};


using SubgraphPtr = std::shared_ptr<Subgraph>;
using SubgraphID = SymbolPtr;


class SubgraphTracker {
public:

    // callback to finalize the subgraph when it is terminated
    using Finalizer = std::function<void(SubgraphPtr)>;

    SubgraphTracker(Finalizer finalizer): m_finalizer(finalizer) {}

    void add_node (NodePtr node, bool belongs) {
        // collect all subgraph ids that input nodes belong to and all dependencies
        Dependencies input_subgraphs;
        Dependencies input_dependencies;
        for(auto input_value: node->input_values()) {
            auto node = input_value.get_node_shared_ptr();
            if(auto id = get_subgraph_id(node)) {
                input_subgraphs.insert(ov::symbol::ancestor_of(id));
            }
            const auto& deps = get_dependencies(node);
            for(auto dep: deps) {
                input_dependencies.insert(ov::symbol::ancestor_of(dep));
            }
        }

        if(belongs) {
            // Below we refuse to merge subgraphs if all of them cannot merge to a single subgraph, this is rough because
            // there are cases when part of the input subgraphs can consume the node and others will come as inputs -- TODO.
            // TODO: leave only those input subgraphs that are not conflicting with other subgraphs nor with any dependencies
            if(input_subgraphs.empty() || intersected(input_subgraphs, input_dependencies)) {   // no input subgraphs || cannot merge all due to cycles
                try_terminate_subgraphs(input_subgraphs, node);

                // start a new subgraph
                auto subgraph_id = new_subgraph();
                add_node_to_subgraph(node, subgraph_id);
                set_subgraph_id(node, subgraph_id);
                input_dependencies.insert(input_subgraphs.begin(), input_subgraphs.end());
            } else {
                auto merged_subgraph_id = std::accumulate(
                    input_subgraphs.begin(),
                    input_subgraphs.end(),
                    *input_subgraphs.begin(),
                    [this](SubgraphID a, SubgraphID b) {
                        merge_subgraphs(a, b);
                        return a;
                    }
                );
                set_subgraph_id(node, merged_subgraph_id);
                add_node_to_subgraph(node, merged_subgraph_id);
            }

        } else {
            try_terminate_subgraphs(input_subgraphs, node);
            set_subgraph_id(node, nullptr);
            input_dependencies.insert(input_subgraphs.begin(), input_subgraphs.end());
        }
        set_dependencies(node, input_dependencies);
    }

    void finalize() {
        for(auto subgraph_record: m_subgraphs) {
            terminate_subgraph(subgraph_record.first);
        }
    }

private:

    std::unordered_map<SubgraphID, SubgraphPtr> m_subgraphs;
    using Dependencies = std::unordered_set<SubgraphID>;
    Finalizer m_finalizer;

    // // Detects if `node` depends on a node from `subgraph` but goes via node that doesn't belongs to `subgraph`
    // bool depends_via_break (NodePtr node, const Subgraph& subgraph);

    SubgraphID new_subgraph() {
        SubgraphID id = std::make_shared<ov::Symbol>();
        m_subgraphs[id] = std::make_shared<Subgraph>();
        return id;
    }

    void add_node_to_subgraph(NodePtr node, SubgraphID id) {
        get_subgraph(id)->nodes.push_back(node);
    }

    void merge_subgraphs(SubgraphID id1, SubgraphID id2) {
        id1 = ov::symbol::ancestor_of(id1);
        id2 = ov::symbol::ancestor_of(id2);
        if (id1 == id2) return;

        auto subgraph1 = get_subgraph(id1);
        auto subgraph2 = get_subgraph(id2);
        subgraph1->merge(*subgraph2);
        m_subgraphs.erase(id1);
        m_subgraphs.erase(id2);
        ov::symbol::set_equal(id1, id2);
        id1 = ov::symbol::ancestor_of(id1);
        m_subgraphs[id1] = subgraph1;
    }

    SubgraphPtr get_subgraph(SubgraphID id) {
        return m_subgraphs.at(ov::symbol::ancestor_of(id));
    }

    // set/get all subgraph ids that contribute to a given node

    const Dependencies& get_dependencies(NodePtr node) {
        return node->get_rt_info().at("__subgraph_dependencies").as<Dependencies>();
    }
    void set_dependencies(NodePtr node, const Dependencies& dependencies) {
        node->get_rt_info()["__subgraph_dependencies"] = dependencies;
    }

    // set/get subgraph id that a give node belongs to

    SubgraphID get_subgraph_id(NodePtr node) {
        auto id = node->get_rt_info().at("__subgraph_id").as<SubgraphID>();
        if(id) {
            id = ov::symbol::ancestor_of(id);
        }
        return id;
    }

    void set_subgraph_id(NodePtr node, SubgraphID id) {
        node->get_rt_info()["__subgraph_id"] = id;
    }

    bool intersected(const Dependencies& a, const Dependencies& b) {
        for(const auto& x: a) {
            if(b.count(x))
                return true;
        }
        return false;
    }

    void terminate_subgraph(SubgraphID id) {
        id = ov::symbol::ancestor_of(id);
        auto subgraph = get_subgraph(id);
        // Build subgraph inputs and outputs
        std::unordered_set<ov::Output<ov::Node>> inputs;
        auto& outputs = subgraph->outputs;
        auto& output_consumers = subgraph->output_consumers;
        for(auto node: subgraph->nodes) {
            for(auto input: node->input_values()) {
                auto input_id = get_subgraph_id(input.get_node_shared_ptr());
                if(!ov::symbol::are_equal(id, input_id)) {
                    inputs.insert(input);
                }
            }
            for(auto output: node->outputs()) {
                const auto& consumers = output.get_target_inputs();
                InputVector external_consumers;
                for(auto consumer: consumers) {
                    auto consumer_id = get_subgraph_id(consumer.get_node()->shared_from_this());
                    if(!ov::symbol::are_equal(id, consumer_id)) {
                        external_consumers.push_back(consumer);
                    }
                }
                bool used_outside = !external_consumers.empty();
                if(used_outside) {
                    outputs.push_back(output);
                    output_consumers.push_back(external_consumers);
                }
            }
        }
        subgraph->inputs.assign(inputs.begin(), inputs.end());
        m_finalizer(subgraph);
    }

    void try_terminate_subgraphs(const Dependencies& subgraphs, NodePtr terminator) {
        // TODO: Terminate subgraphs earlier when all terminating nodes are known
        // TODO: try to merge subgraphs if they are being terminated simultaniously
    }
};

// This pass find marked with a special flag group of nodes and collapse each group to a single MLIR function
NodePtr ngraph_to_mlir_op(MLIRContext* context, SubgraphPtr subgraph) {

    mlir::OwningOpRef<mlir::ModuleOp> module;

    module = ngraph_to_mlir(context, subgraph->inputs, subgraph->nodes, subgraph->outputs);

    OVOutputTypes output_types;
    for (size_t i = 0; i < subgraph->outputs.size(); ++i) {
        output_types.push_back(
            std::make_tuple(subgraph->outputs[i].get_element_type(), subgraph->outputs[i].get_partial_shape()));
    }
    return std::make_shared<MLIROp>(
        subgraph->inputs,
        std::make_shared<MLIREvaluate>(std::move(module)),
        output_types
    );
};


const std::string& subgraph_mark() {
    static const std::string mark = "__subgraph_mlir_mark";
    return mark;
}

void set_subgraph_mark(NodePtr node) {
    node->get_rt_info()[subgraph_mark()];
}

bool get_subgraph_mark(NodePtr node) {
    return node->get_rt_info().count(subgraph_mark());
}

class MarkPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkPattern", "0");
    MarkPattern(NodePtr pattern) {
        auto callback = [](ov::pass::pattern::Matcher& m) {
            // TODO: support multi-node patterns marking
            auto node = m.get_match_root();
            set_subgraph_mark(node);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "MarkPattern");
        register_matcher(m, callback);
    }
};


void replace_subgraph(SubgraphPtr subgraph, NodePtr node) {
    const auto& output_consumers = subgraph->output_consumers;
    assert(output_consumers.size() == node->get_output_size());
    for(size_t i = 0; i < node->get_output_size(); ++i) {
        auto replacement = node->output(i);
        for(auto consumer: output_consumers[i]) {
            consumer.replace_source_output(replacement);
        }
    }
}


class Partitioner : public ov::pass::ModelPass {
    MLIRContext* context;
public:
    OPENVINO_RTTI("Partitioner");

    Partitioner(MLIRContext* context) : context(context) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        SubgraphTracker tracker([this](SubgraphPtr subgraph) {
                auto mlir_op = ngraph_to_mlir_op(context, subgraph);
                replace_subgraph(subgraph, mlir_op);
                std::cerr << "Created MLIR op: " << mlir_op << "\n";
            }
        );
        for(auto node: model->get_ordered_ops()) {
            tracker.add_node(node, get_subgraph_mark(node));
        }
        tracker.finalize();
    }
};


bool elementwise_f32_binary_no_broadcast_predicate(const ov::Output<ov::Node>& output) {
    if(output.get_element_type() != ov::element::f32) {
        return false;
    }
    // Check if implicit broadcast is possible, reject in this case
    // Relies on symbolic information -- register SymbolicPropagation before applying this pattern
    auto input_shape_a = output.get_node_shared_ptr()->get_input_partial_shape(0);
    auto input_shape_b = output.get_node_shared_ptr()->get_input_partial_shape(1);
    auto output_shape = output.get_partial_shape();
    if(output_shape.rank().is_dynamic() || input_shape_a.rank().is_dynamic() || input_shape_b.rank().is_dynamic()) {
        return false;
    }
    if(output_shape.rank().get_length() != input_shape_a.rank().get_length() || output_shape.rank().get_length() != input_shape_b.rank().get_length()) {
        return false;
    }

    for(size_t i = 0; i < output_shape.size(); ++i) {
        if(output_shape[i] != input_shape_a[i] || output_shape[i] != input_shape_b[i]) {
            return false;
        }
        if(!ov::symbol::are_equal(output_shape[i].get_symbol(), input_shape_a[i].get_symbol()) || !ov::symbol::are_equal(output_shape[i].get_symbol(), input_shape_b[i].get_symbol())) {
            return false;
        }
    }

    return true;
}


template <typename Op>
NodePtr elementwise_f32_binary_no_broadcast() {
    using namespace ov::pass::pattern;
    return wrap_type<Op>({any_input(), any_input()}, elementwise_f32_binary_no_broadcast_predicate);
}


void injectMLIR(std::shared_ptr<ov::Model> model, MLIRContext* context) {
    ov::pass::Manager manager;
    using namespace ov::op;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::SymbolicPropagation>();
    manager.register_pass<MarkPattern>(elementwise_f32_binary_no_broadcast<v1::Add>());
    manager.register_pass<MarkPattern>(elementwise_f32_binary_no_broadcast<v1::Subtract>());
    manager.register_pass<MarkPattern>(elementwise_f32_binary_no_broadcast<v1::Multiply>());
    manager.register_pass<MarkPattern>(elementwise_f32_binary_no_broadcast<v1::Divide>());
    manager.register_pass<Partitioner>(context);
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
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

} // namespace

void ov::pass::transformMLIR(std::shared_ptr<ov::Model> model) {
    injectMLIR(model, get_shared_mlir_context());
}
