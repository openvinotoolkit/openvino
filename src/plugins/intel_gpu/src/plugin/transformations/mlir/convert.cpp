// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interface/convert.hpp"

#include <algorithm>
#include <functional>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/abs.hpp>
#include <openvino/op/ceiling.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/log.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/tanh.hpp>
#include <openvino/op/reduce_max.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reduce_min.hpp>
#include <openvino/op/reduce_prod.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/util/env_util.hpp>
#include <unordered_map>
#include <unordered_set>

// TODO: Prune unused headers -- it's hard to understand needed ones
#include "graph_converter.hpp"
#include "common/convert_common.hpp"
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
#include "mlir/Dialect/Shape/IR/Shape.h"
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

#include "gc/Transforms/Passes.h"

#include "intel_gpu/op/mlir_op.hpp"
#include "mlir_evaluate.hpp"
#include "conversion/patterns.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/symbol.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "subgraph_tracker.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace {

using namespace mlir;
using namespace ov::intel_gpu::mlir;

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

// Erases input function args that have no uses.
void dropUnusedInputArgs(mlir::func::FuncOp func, size_t numInputs, SmallVector<size_t>& kept) {
    llvm::BitVector toErase(func.getNumArguments());
    for (size_t i = 0; i < numInputs; ++i) {
        auto arg = func.getArgument(i);
        if (arg.use_empty()) {
            toErase.set(i);
            continue;
        }
        if (arg.hasOneUse()) {
            auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(*arg.user_begin());
            if (toTensor && toTensor.getResult().use_empty()) {
                toTensor.erase();
                toErase.set(i);
                continue;
            }
        }
        kept.push_back(i);
    }
    func.eraseArguments(toErase);
}

mlir::OwningOpRef<mlir::ModuleOp> ngraph_to_mlir(MLIRContext* context,
                                                 const ov::OutputVector& inputs,
                                                 const ov::NodeVector& nodes,
                                                 const ov::OutputVector& outputs,
                                                 SmallVector<size_t>& keptInputIndices,
                                                 const std::string& function_name = "entry") {
    // Split inputs: splat constants are inlined, runtime inputs become function args.
    ov::OutputVector runtime_inputs;
    SmallVector<bool> is_constant(inputs.size(), false);
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto* c = ov::as_type<ov::op::v0::Constant>(inputs[i].get_node());
        if (c && c->get_all_data_elements_bitwise_identical()) {
            is_constant[i] = true;
        } else {
            runtime_inputs.push_back(inputs[i]);
        }
    }

    auto inputTypes = tensorsToMemRefs(get_types_for_values(context, runtime_inputs));
    auto outputTypes = tensorsToMemRefs(get_types_for_values(context, outputs));

    const auto moduleLoc = createLayerLocation(context, "module", "Module");
    auto module = mlir::ModuleOp::create(moduleLoc, StringRef("fragment_name"));

    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody() /* TODO: building log here */);

    const auto funcLoc = createLayerLocation(context, "entry", "Func");
    auto memref_args = inputTypes;
    memref_args.append(outputTypes);
    const auto funcType = mlir::FunctionType::get(context, ArrayRef(memref_args), ArrayRef(SmallVector<mlir::Type>()));
    auto trunc = function_name.rfind('#');
    auto name = trunc == std::string::npos ? function_name : function_name.substr(0, trunc);
    auto func = mlir::func::FuncOp::create(moduleBuilder, funcLoc, name, funcType);
    auto block_builder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock() /* TODO: Add logger here */);

    GraphConverter graph_converter(context, &block_builder);

    for (size_t i = 0, r = 0; i < inputs.size(); ++i) {
        auto loc = createLocation(context, inputs[i].get_node_shared_ptr());
        if (is_constant[i]) {
            auto* cst = ov::as_type<ov::op::v0::Constant>(inputs[i].get_node());    
            graph_converter.nodeOutputMap.emplace(inputs[i], getConstant(block_builder, cst, loc));
        } else {
            auto funcInputVal = func.getArgument(r++);
            // transition from memref enclosure to tensor interior
            auto ranked = mlir::dyn_cast<mlir::MemRefType>(funcInputVal.getType());
            auto tensorTy = mlir::RankedTensorType::get(ranked.getShape(), ranked.getElementType());
            auto tensor = block_builder.create<bufferization::ToTensorOp>(
                loc, tensorTy, funcInputVal, /*restrict = */ true, /*writable=*/ true);
            graph_converter.nodeOutputMap.emplace(inputs[i], tensor);

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
                        if(dim.is_dynamic() && !graph_converter.dimension_map.count(symbol)) {
                            auto dimSize = block_builder.create<tensor::DimOp>(loc, tensor, j);
                            graph_converter.dimension_map[symbol] = dimSize;
                        }
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto node = nodes[i];
        graph_converter.convert(node);
    }

    SmallVector<mlir::Value> funcOutputs;
    funcOutputs.reserve(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = graph_converter.nodeOutputMap.at(outputs[i]);
        auto memref = func.getArgument(i + runtime_inputs.size());
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
    SmallVector<size_t> keptRuntimeIndices;
    dropUnusedInputArgs(func, runtime_inputs.size(), keptRuntimeIndices);
    auto runtime_to_original = llvm::map_to_vector(
        llvm::make_filter_range(llvm::enumerate(is_constant), [](const auto& p) { return !p.value(); }),
        [](const auto& p) { return p.index(); });
    llvm::transform(keptRuntimeIndices, std::back_inserter(keptInputIndices),
        [&](size_t i) { return runtime_to_original[i]; });
    return module;
}

// This pass converts a group of nodes into a single MLIROp
NodePtr ngraph_to_mlir_op(MLIRContext* context,
                          SubgraphPtr subgraph,
                          std::shared_ptr<ov::EvaluationContext> loweringContext) {
    SmallVector<size_t> keptInputIndices;
    mlir::OwningOpRef<mlir::ModuleOp> module =
        ngraph_to_mlir(context, subgraph->inputs, subgraph->nodes, subgraph->outputs, keptInputIndices,
                       subgraph->function_name);

    ov::OutputVector inputs;
    inputs.reserve(keptInputIndices.size());
    for (size_t idx : keptInputIndices) {
        inputs.push_back(subgraph->inputs[idx]);
    }
    using Index = ov::intel_gpu::op::DimensionsMap::value_type::value_type;
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
                    OPENVINO_MLIR_DEBUG_PRINT("Lost equality constraint for dimensions in output " << input << ".");
                    OPENVINO_MLIR_DEBUG_PRINT("If the constraint is violated in runtime it will result in the undefined behaviour.");
                }
            }
        }
    }

    std::tuple<size_t, size_t> empty(-1, -1);
    const auto& outputs = subgraph->outputs;
    ov::intel_gpu::op::OVOutputTypes output_types;
    ov::intel_gpu::op::DimensionsMap output_map;
    output_map.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = outputs[i];
        auto shape = output.get_partial_shape();
        output_types.push_back(
            std::make_tuple(output.get_element_type(), shape));
        ov::intel_gpu::op::DimensionsMap::value_type dm;
        dm.reserve(shape.size());
        for (size_t j = 0; j < shape.size(); ++j) {
            auto dim = shape[j];
            if (dim.is_dynamic())
                assert(input_map.count(ov::symbol::ancestor_of(dim.get_symbol())) && "Input map is missing a symbol for dynamic dim");
            dm.push_back(dim.is_dynamic() ? input_map.at(ov::symbol::ancestor_of(dim.get_symbol())) : empty);
        }
        output_map.emplace_back(dm);
    }
    return std::make_shared<ov::intel_gpu::op::MLIROp>(
        inputs,
        std::make_shared<MLIREvaluateGcGPU>(std::move(module), loweringContext),
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

// Marks matched subgraphs with a custom function name so the
// Partitioner groups them into a dedicated MLIR function.
// The patterns are specified with the env var:
//   OV_MLIR_PATTERNS="name1=Type1,Type2;name2=Type3,Type4,...".
class PatternMatcher : public ov::pass::ModelPass {
    struct NamedPattern {
        std::string name;
        std::vector<std::string> types;
    };

    const std::vector<NamedPattern> patterns = []() {
        std::vector<NamedPattern> patterns;
        const auto& spec = ov::util::getenv_string("OV_MLIR_PATTERNS");
        size_t pos = 0;
        while (pos < spec.size()) {
            auto sep = spec.find(';', pos);
            auto entry = spec.substr(pos, sep == std::string::npos ? std::string::npos : sep - pos);
            pos = sep == std::string::npos ? spec.size() : sep + 1;

            auto eq = entry.find('=');
            if (eq == std::string::npos)
                continue;
            NamedPattern p{entry.substr(0, eq), {}};
            for (size_t tp = eq + 1; tp < entry.size();) {
                auto comma = entry.find(',', tp);
                auto type = entry.substr(tp, comma == std::string::npos ? std::string::npos : comma - tp);
                if (!type.empty())
                    p.types.push_back(type);
                tp = comma == std::string::npos ? entry.size() : comma + 1;
            }
            if (!p.name.empty() && !p.types.empty())
                patterns.push_back(std::move(p));
        }
        return patterns;
    }();

public:
    OPENVINO_RTTI("PatternMatcher");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        if (patterns.empty())
            return false;

        auto ordered_ops = model->get_ordered_ops();
        auto filter = std::remove_if(ordered_ops.begin(), ordered_ops.end(), [](const auto& n) {
            static std::string skip[] = {"Constant", "Parameter", "Result"};
            for (const auto& s : skip) {
                if (s == n->get_type_info().name)
                    return true;
            }
            return false;
        });
        ordered_ops.erase(filter, ordered_ops.end());

        auto print_chain = [&](const char* msg, size_t offset, size_t count) {
            std::string chain;
            for (size_t i = offset; i < offset + count; ++i) {
                chain += ordered_ops[i]->get_type_info().name;
                chain += ",";
            }
            if (!chain.empty()) {
                chain.pop_back();
                OPENVINO_MLIR_DEBUG_PRINT(msg << chain);
            }
        };

        if (::ov::intel_gpu::mlir::is_debug())
            print_chain("Matching model: ", 0, ordered_ops.size());

        bool changed = false;
        std::unordered_set<ov::Node*> matched;
        // Per-pattern match counter: each match gets a unique name (name, name1, name2, ...) so
        // repeated matches stay separate single-output subgraphs instead of being force-joined by name.
        std::unordered_map<std::string, int> match_count;
        for (size_t i = 0, count = ordered_ops.size(); i < count; ++i) {
            auto node = ordered_ops[i];
            for (const auto& p : patterns) {
                if (p.types.size() > count - i || p.types.front() != node->get_type_info().name || !has_subgraph_mark(node))
                    continue;
                bool all_matches = true;
                size_t len = p.types.size();
                for (size_t n = i + 1, t = 1; t < len; ++n, ++t) {
                    if (p.types[t] != ordered_ops[n]->get_type_info().name || !has_subgraph_mark(ordered_ops[n])) {
                        all_matches = false;
                        break;
                    }
                }
                // Connectivity check
                if (all_matches && len > 1) {
                    std::unordered_set<ov::Node*> nodes;
                    for (size_t t = 0; t < len; ++t)
                        nodes.insert(ordered_ops[i + t].get());
                    std::unordered_set<ov::Node*> seen{ordered_ops[i].get()};
                    std::vector<ov::Node*> stack{ordered_ops[i].get()};
                    auto visit = [&](ov::Node* nb) {
                        if (nodes.count(nb) && seen.insert(nb).second)
                            stack.push_back(nb);
                    };
                    while (!stack.empty()) {
                        auto* cur = stack.back();
                        stack.pop_back();
                        for (auto& in : cur->input_values())
                            visit(in.get_node());
                        for (auto& out : cur->outputs())
                            for (auto& ti : out.get_target_inputs())
                                visit(ti.get_node());
                    }
                    all_matches = seen.size() == len;
                }
                if (all_matches) {
                    int n = match_count[p.name]++;
                    std::string name = n ? p.name + "#" + std::to_string(n) : p.name;
                    if (::ov::intel_gpu::mlir::is_debug()) {
                        std::string msg = "Matched pattern " + name + "=";
                        print_chain(msg.c_str(), i, len);
                    }
                    for (size_t t = 0; t < len; ++t, ++i) {
                        set_subgraph_mark(ordered_ops[i], name);
                        matched.insert(ordered_ops[i].get());
                    }
                    changed = true;
                    --i;
                    break;
                }
            }
        }

        // Clear marks on nodes not matched by any pattern - they will not be converted to MLIR.
        for (auto node : ordered_ops) {
            if (matched.count(node.get()) == 0) {
                set_subgraph_mark(node, "");
                changed = true;
            }
        }

        return changed;
    }
};

class Partitioner : public ov::pass::ModelPass {
    MLIRContext* context;
    std::shared_ptr<ov::EvaluationContext> loweringContext;
public:
    OPENVINO_RTTI("Partitioner");

    Partitioner(MLIRContext* context, std::shared_ptr<ov::EvaluationContext> loweringContext) :
        context(context),
        loweringContext(loweringContext)
    {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        SubgraphTracker tracker([this](SubgraphPtr subgraph) {
            auto mlir_op = ngraph_to_mlir_op(context, subgraph, loweringContext);
            replace_subgraph(subgraph, mlir_op);
            OPENVINO_MLIR_DEBUG_PRINT("Created MLIR op: " << mlir_op);
        });
        for (auto node : model->get_ordered_ops()) {
            tracker.add_node(node, get_subgraph_mark(node));
        }
        tracker.finalize();
        return true;
    }
};


void injectMLIR(std::shared_ptr<ov::Model> model,
                MLIRContext* context,
                std::shared_ptr<ov::EvaluationContext> loweringContext) {
    ov::pass::Manager manager;
    using namespace ov::op;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::SymbolicPropagation>();
    manager.register_pass<SDPAPattern>();
    manager.register_pass<BinaryEltwisePattern<v1::Add, linalg::AddOp>>();
    manager.register_pass<BinaryEltwisePattern<v1::Subtract, linalg::SubOp>>();
    manager.register_pass<BinaryEltwisePattern<v1::Multiply, linalg::MulOp>>();
    manager.register_pass<BinaryEltwisePattern<v1::Divide, linalg::DivOp>>();
    manager.register_pass<BinaryEltwisePattern<v1::Power, linalg::PowFOp>>();
    manager.register_pass<ReducePattern<v1::ReduceMax>>();
    manager.register_pass<ReducePattern<v1::ReduceMean>>();
    manager.register_pass<ReducePattern<v1::ReduceMin>>();
    manager.register_pass<ReducePattern<v1::ReduceProd>>();
    manager.register_pass<ReducePattern<v1::ReduceSum>>();
    manager.register_pass<ReshapePattern>();
    manager.register_pass<ConcatPattern>();
    manager.register_pass<ReluPattern>();
    manager.register_pass<FloorPattern>();
    manager.register_pass<UnaryEltwisePattern<v0::Abs, linalg::AbsOp>>();
    manager.register_pass<UnaryEltwisePattern<v0::Ceiling, linalg::CeilOp>>();
    manager.register_pass<UnaryEltwisePattern<v0::Exp, linalg::ExpOp>>();
    manager.register_pass<UnaryEltwisePattern<v0::Log, linalg::LogOp>>();
    manager.register_pass<UnaryEltwisePattern<v0::Negative, linalg::NegFOp>>();
    manager.register_pass<UnaryEltwisePattern<v0::Sqrt, linalg::SqrtOp>>();
    manager.register_pass<UnaryEltwisePattern<v0::Tanh, linalg::TanhOp>>();
    manager.register_pass<GatherPattern>();
    manager.register_pass<ShapeOfPattern>();
    manager.register_pass<SlicePattern>();
    manager.register_pass<SqueezePattern>();
    manager.register_pass<TransposePattern>();
    manager.register_pass<UnsqueezePattern>();
    manager.register_pass<MatMulPattern>();
    manager.register_pass<PatternMatcher>();
    manager.register_pass<Partitioner>(context, loweringContext);
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}

MLIRContext* get_shared_mlir_context() {
    static auto context = [] {
        auto ctx = std::make_unique<MLIRContext>(gc::getDialectRegistry());
        ctx->loadAllAvailableDialects();
        return ctx;
    }();
    return context.get();
}

} // namespace

void ov::intel_gpu::mlir::transformMLIR(std::shared_ptr<ov::Model> model,
                                        std::shared_ptr<ov::EvaluationContext> loweringContext) {
    injectMLIR(model, get_shared_mlir_context(), loweringContext);
}
