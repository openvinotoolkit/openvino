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
#include "openvino/core/dimension.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations_visibility.hpp"
#include "openvino/core/symbol.hpp"

#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

#include "mlir_op.hpp"
#include "convert_common.hpp"
#include "subgraph_tracker.hpp"


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

class ConversionContext {
public:
    using Convertor = std::function<void(ConversionContext&, NodePtr)>;
    using NodeOutputMap = std::map<ov::Output<ov::Node>, mlir::Value>;

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
