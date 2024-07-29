// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/mlir/convert.hpp"

#include <algorithm>
#include <functional>
#include <openvino/op/add.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/util/env_util.hpp>
#include <unordered_map>

// TODO: Prune unused headers -- it's hard to understand needed ones
#include "conversion_context.hpp"
#include "convert_common.hpp"
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
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/GPU/Utils.h"
#endif

#include "mlir_op.hpp"
#include "op/matmul.hpp"
#include "op/relu.hpp"
#include "op/binary_eltwise.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/symbol.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "subgraph_tracker.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations_visibility.hpp"

namespace {

using namespace mlir;
using namespace ov::mlir;

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

        // FIXME: Avoid pre-population of dimension_map, take dimension values only if needed
        auto input_shape = inputs[i].get_partial_shape();
        auto input_rank = input_shape.rank();
        if(input_rank.is_static()) {
            for(size_t j = 0; j < input_rank.get_length(); ++j) {
                auto dim = input_shape[j];
                if(dim.is_dynamic()) {
                    auto symbol = dim.get_symbol();
                    assert(symbol);
                    symbol = ov::symbol::ancestor_of(symbol);
                    if(dim.is_dynamic() && !conversion_context.dimension_map.count(symbol)) {
                        auto dimSize = block_builder.create<tensor::DimOp>(loc, tensor, j);
                        conversion_context.dimension_map[symbol] = dimSize;
                    }
                }
            }
        }
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


// This pass converts a group of nodes into a single MLIROp
NodePtr ngraph_to_mlir_op(MLIRContext* context, SubgraphPtr subgraph, bool tpp_mlir_enabled) {
    mlir::OwningOpRef<mlir::ModuleOp> module = ngraph_to_mlir(context, subgraph->inputs, subgraph->nodes, subgraph->outputs);

    const auto& inputs = subgraph->inputs;
    using Index = DimensionsMap::value_type::value_type;
    std::map<SymbolPtr, Index> input_map;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input = inputs[i];
        auto shape = input.get_partial_shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            auto dim = shape[j];
            if(shape[j].is_dynamic()) {
                auto symbol = ov::symbol::ancestor_of(dim.get_symbol());
                if(0 == input_map.count(symbol)) {
                    input_map[symbol] = Index(i, j);
                } else {
                    OPENVINO_MLIR_DEBUG_PRINT(
                        "[ DEBUG ] Lost equality constraint for dimensions in output " << input << ".\n" <<
                        "          If the constraint is violated in runtime it will result in the undefined behaviour.\n");
                }
            }
        }
    }

    std::tuple<size_t, size_t> empty(-1, -1);
    const auto& outputs = subgraph->outputs;
    OVOutputTypes output_types;
    DimensionsMap output_map;
    output_map.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = outputs[i];
        auto shape = output.get_partial_shape();
        output_types.push_back(
            std::make_tuple(output.get_element_type(), shape));
        DimensionsMap::value_type dm;
        dm.reserve(shape.size());
        for (size_t j = 0; j < shape.size(); ++j) {
            auto dim = shape[j];
            dm.push_back(dim.is_dynamic() ? input_map.at(ov::symbol::ancestor_of(dim.get_symbol())) : empty);
        }
        output_map.emplace_back(dm);
    }
    return std::make_shared<MLIROp>(
        subgraph->inputs,
        std::make_shared<MLIREvaluate>(std::move(module), tpp_mlir_enabled),
        output_types,
        output_map
    );
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
    bool tpp_mlir_enabled;
public:
    OPENVINO_RTTI("Partitioner");

    Partitioner(MLIRContext* context, bool tpp_mlir_enabled) :
        context(context),
        tpp_mlir_enabled(tpp_mlir_enabled)
    {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        SubgraphTracker tracker([this](SubgraphPtr subgraph) {
                auto mlir_op = ngraph_to_mlir_op(context, subgraph, tpp_mlir_enabled);
                replace_subgraph(subgraph, mlir_op);
                OPENVINO_MLIR_DEBUG_PRINT("Created MLIR op: " << mlir_op << "\n");
            }
        );
        for(auto node: model->get_ordered_ops()) {
            tracker.add_node(node, get_subgraph_mark(node));
        }
        tracker.finalize();
        return true;
    }
};


void injectMLIR(std::shared_ptr<ov::Model> model, MLIRContext* context, bool tpp_mlir_enabled) {
    ov::pass::Manager manager;
    using namespace ov::op;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::SymbolicPropagation>();
    manager.register_pass<BinaryEltwisePattern<v1::Add, linalg::AddOp>>();
    manager.register_pass<BinaryEltwisePattern<v1::Subtract, linalg::SubOp>>();
    manager.register_pass<BinaryEltwisePattern<v1::Multiply, linalg::MulOp>>();
    manager.register_pass<BinaryEltwisePattern<v1::Divide, linalg::DivOp>>();
    manager.register_pass<ReluPattern>();
    manager.register_pass<MatMulPattern>();
    manager.register_pass<Partitioner>(context, tpp_mlir_enabled);
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}


MLIRContext* get_shared_mlir_context(bool tpp_mlir_enabled_current) {
    // Gives MLIRContext instance shared for entire OV process and initialized once upon the initial request
    // FIXME: Bind with OpenVINO lifetime in the sutable class instead of dirty tricking with static lifetime

    static std::shared_ptr<MLIRContext> context;
    static bool tpp_mlir_enabled = tpp_mlir_enabled_current;

    if(context) {
        if(tpp_mlir_enabled_current != tpp_mlir_enabled) {
            OPENVINO_MLIR_DEBUG_PRINT("[ DEBUG ] Switched TPP mode, reinitialize MLIR context\n");
            tpp_mlir_enabled = tpp_mlir_enabled_current;
            context.reset();
        }
    }

    if (!context) {

        // Initialize the LLVM machinery
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        OPENVINO_MLIR_DEBUG_PRINT("[ DEBUG ] Using TPP_MLIR: ");
        if(tpp_mlir_enabled) {
            OPENVINO_MLIR_DEBUG_PRINT("YES\n");
            // Initialize GPU-related LLVM machinery
            #ifdef TPP_MLIR
                tpp::initializeGpuTargets();
            #endif
        } else {
            OPENVINO_MLIR_DEBUG_PRINT("NO\n");
        }

        // Add the following to include *all* MLIR Core dialects, or selectively
        // include what you need like above. You only need to register dialects that
        // will be *parsed* by the tool, not the one generated
        DialectRegistry registry;
        if(tpp_mlir_enabled) {
            #ifdef TPP_MLIR
                registry.insert<mlir::xsmm::XsmmDialect>();
                registry.insert<mlir::check::CheckDialect>();
                registry.insert<mlir::perf::PerfDialect>();
            #endif
        }

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
    if(util::getenv_bool("OV_MLIR", true)) {
        bool tpp_mlir_default =
            #ifdef TPP_MLIR
                true;
            #else
                false;
            #endif
        bool tpp_mlir_enabled = util::getenv_bool("OV_MLIR_TPP", tpp_mlir_default);
        #ifndef TPP_MLIR
            OPENVINO_ASSERT(!tpp_mlir_enabled,
                "[ ERROR ] OpenVINO wasn't compiled with TPP_MLIR support, "
                "but OV_MLIR_TPP environment variable is set to enable it.");
        #endif

        injectMLIR(model, get_shared_mlir_context(tpp_mlir_enabled), tpp_mlir_enabled);
    }
}
