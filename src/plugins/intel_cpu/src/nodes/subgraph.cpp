// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph.h"

#include <ie_parallel.hpp>

#include <vector>
#include <algorithm>
#include <array>
#include <tuple>

#include <dnnl_debug.h>
#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/rt_info.hpp>
#include <ie_ngraph_utils.hpp>

#include <snippets/op/subgraph.hpp>
#include "snippets/pass/matmul_to_brgemm.hpp"
#include "utils/cpu_utils.hpp"
#include "emitters/x64/cpu_generator.hpp"
#include "transformations/snippets/x64/pass/lowered/fuse_load_store_and_convert.hpp"
#include "transformations/snippets/x64/pass/lowered/brgemm_blocking.hpp"
#include "transformations/snippets/x64/pass/mul_add_to_fma.hpp"
#include "transformations/snippets/x64/pass/brgemm_to_brgemm_cpu.hpp"
#include "transformations/snippets/x64/pass/remove_converts.hpp"
#include "transformations/snippets/x64/pass/enforce_precision.hpp"
#include "transformations/snippets/x64/pass/set_brgemm_cpu_blocking_params.hpp"
#include "transformations/cpu_opset/common/pass/convert_to_swish_cpu.hpp"
#include "transformations/defs.hpp"
#include "shape_inference/custom/subgraph.hpp"

using namespace InferenceEngine;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {

Snippet::Snippet(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, SnippetShapeInferFactory(this)) {
    host_isa = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ?
        dnnl::impl::cpu::x64::avx512_core : dnnl::impl::cpu::x64::avx2;
    original_snippet = ov::as_type_ptr<snippets::op::Subgraph>(op);
    if (!original_snippet) {
        IE_THROW(NotImplemented) << "Node is not an instance of snippets::op::Subgraph";
    }
}

void Snippet::copy_snippet() {
    ov::OutputVector subgraph_node_inputs;
    for (const auto &input : original_snippet->input_values()) {
        auto new_input = std::make_shared<ov::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
        subgraph_node_inputs.push_back(new_input);
    }
    std::shared_ptr<ov::Model> new_body = original_snippet->body_ptr()->clone();
    snippet = std::make_shared<snippets::op::Subgraph>(subgraph_node_inputs, new_body);
    ov::copy_runtime_info(original_snippet, snippet);
    snippet->set_friendly_name(original_snippet->get_friendly_name());
#if defined(OPENVINO_ARCH_X86_64)
    snippet->set_generator(std::make_shared<CPUGenerator>(host_isa));
    isa_num_lanes =  snippet->get_generator()->get_target_machine()->get_lanes();
#else
    IE_THROW(NotImplemented) << "CPU plugin: code-generation is not supported on non-x64 platforms";
#endif // OPENVINO_ARCH_X86_64
}

void Snippet::initSupportedPrimitiveDescriptors() {
    copy_snippet();
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::set<Precision> supportedPrecisions = { Precision::FP32, Precision::I32, Precision::BF16, Precision::FP16, Precision::I8, Precision::U8 };

    bool dimRanksAreEqual = true;
    for (size_t i = 0; dimRanksAreEqual && i < inputShapes.size(); i++) {
        for (size_t j = 0; dimRanksAreEqual && j < outputShapes.size(); j++) {
            if (inputShapes[i].getRank() != outputShapes[j].getRank())
                dimRanksAreEqual = false;
        }
    }

    const size_t ndims = outputShapes[0].getRank();
    // Domain sensitive operations support only Planar layout
    const bool isOnlyPlanarApplicable = snippet->has_domain_sensitive_ops();
    const bool isChannelsFirstApplicable = dnnl::impl::utils::one_of(ndims, 1u, 2u, 3u, 4u, 5u) && dimRanksAreEqual && !isOnlyPlanarApplicable;
    // Todo: Snippets currently don't support per-channel broadcasting of Blocked descriptors because
    //  canonicalization can't distinguish between <N, C, H, W, c> and <N, C, D, H, W> cases.
    //  See snippets::op::Subgraph::canonicalize for details.
    bool isBlockedApplicable = dnnl::impl::utils::one_of(ndims,  4u, 5u) && dimRanksAreEqual && !isOnlyPlanarApplicable;

    for (const auto& inShape : inputShapes) {
        if (isDynamic && inShape.getRank() != 1)
            isBlockedApplicable = isBlockedApplicable && inShape.getMinDims()[1] != Shape::UNDEFINED_DIM && inShape.getMinDims()[1] > 1;
    }

    enum LayoutType {
        Planar,
        ChannelsFirst,
        Blocked
    };
    auto initDesc = [&] (LayoutType lt) -> NodeDesc {
        auto createMemoryDesc = [lt](const Shape &shape, Precision prc, size_t offset) -> std::shared_ptr<CpuBlockedMemoryDesc> {
            const auto &dims = shape.getDims();
            if (lt == ChannelsFirst && shape.getRank() != 1) {
                auto ndims = shape.getRank();
                VectorDims order(ndims);
                std::iota(order.begin(), order.end(), 0);
                if (ndims > 1) {
                    order.erase(order.begin() + 1);
                    order.push_back(1);
                }

                VectorDims blocks(ndims);
                for (size_t i = 0; i < order.size(); i++) {
                    blocks[i] = dims[order[i]];
                }

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            } else if (lt == Blocked && shape.getRank() != 1 && (shape.getMinDims()[1] != Shape::UNDEFINED_DIM && shape.getMinDims()[1] > 1)) {
                size_t blockSize = mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 16 : 8;

                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                blocks[1] = dims[1] != Shape::UNDEFINED_DIM ? div_up(blocks[1], blockSize) : Shape::UNDEFINED_DIM;
                blocks.push_back(blockSize);
                order.push_back(1);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            } else {
                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            }
        };

        size_t offset = 0;
        NodeConfig config;
        config.inConfs.resize(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); i++) {
            const auto originalInputPrecision = getOriginalInputPrecisionAtPort(i);
            const auto precision = ((originalInputPrecision == InferenceEngine::Precision::FP32) &&
                                     context->getConfig().inferencePrecision == ov::element::bf16 &&
                                     snippet->has_domain_sensitive_ops()) ?
                static_cast<InferenceEngine::Precision>(InferenceEngine::Precision::BF16) :
                originalInputPrecision;
            if (supportedPrecisions.count(precision) == 0)
                IE_THROW() << "Subgraph node with name `" << getName() << "` doesn't support " << precision << " precision.";

            const auto equalPrecisions = getOriginalOutputPrecisions().size() == 1 &&
                    precision == getOriginalOutputPrecisionAtPort(0);

            BlockedMemoryDesc::CmpMask inputMask = BlockedMemoryDesc::SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace((!i && canBeInPlace() && equalPrecisions) ? 0 : -1);
            portConfig.constant(false);
            if (inputShapes[i].getDims()[0] == 1) {
                inputMask.reset(0); // accepts any stride on batch axis
            }
            portConfig.setMemDesc(createMemoryDesc(inputShapes[i], precision, offset), inputMask);
            config.inConfs[i] = portConfig;
        }
        config.outConfs.resize(outputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); i++) {
            auto precision = getOriginalOutputPrecisionAtPort(i);
            if (supportedPrecisions.count(precision) == 0)
                IE_THROW() << "Subgraph node with name `" << getName() << "` doesn't support " << precision << " precision.";

            BlockedMemoryDesc::CmpMask outputMask = BlockedMemoryDesc::SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace(-1);
            portConfig.constant(false);
            if (outputShapes[i].getDims()[0] == 1) {
                outputMask.reset(0); // accepts any stride on batch axis
            }
            portConfig.setMemDesc(createMemoryDesc(outputShapes[i], precision, offset), outputMask);
            config.outConfs[i] = portConfig;
        }

        impl_desc_type impl_type = impl_desc_type::unknown;
        if (mayiuse(x64::avx512_core)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (mayiuse(x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        }
        return {config, impl_type};
    };

    if (isChannelsFirstApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
    if (isBlockedApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
    supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));
}

void Snippet::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getImplPriority(), true);
}
InferenceEngine::Precision Snippet::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated && !parentEdge->getParent()->isConstant()) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->getDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

bool Snippet::optimizeExecDomain(std::vector<VectorDims>& inputShapes, std::vector<VectorDims>& outputShapes,
                                 VectorDims &domain, size_t& TileRank) const {
    const size_t minimalConcurrency = parallel_get_max_threads();
    const size_t minimalJitWorkAmount = 256;
    const size_t ds = domain.size();
    if ( ds <= 2 || // not enough dimensions to collapse
         domain[ds-1] >= minimalJitWorkAmount || // There is enough work for 1D Tiles, no need to collapse
         domain[ds-1] * domain[ds-2] >= fullWorkAmount / minimalConcurrency) // There won't be enough work for every thread (even one iter) if we collapse
        return false;
    auto findDimsToCollapse = [&]() {
        auto collapseLastDims = [](VectorDims& dims, size_t dimsToCollapse) {
            if (dimsToCollapse >= dims.size() - 1)
                IE_THROW() << "Got invalid number of dims to collapse. Expected < " << dims.size() - 1 << " got " << dimsToCollapse;
            for (int i = dims.size() - 2; i > static_cast<int>(dims.size() - dimsToCollapse - 2); i--) {
                dims[dims.size() - 1] *= dims[i];
            }

            for (int i = dims.size() - 2; i >= static_cast<int>(dimsToCollapse); i--) {
                dims[i] = dims[i - dimsToCollapse];
            }

            for (int i = dimsToCollapse - 1; i >= 0; i--) {
                dims[i] = 1;
            }
        };
        int collapsedDims = 0;
        size_t currentJitWorkAmount = domain[domain.size() - 1];
        while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount) {
            if (static_cast<int>(domain.size()) - collapsedDims - 2 < 0)
                break;

            bool canCollapse = true;
            for (size_t i = 0; i < inputShapes.size(); i++) {
                const size_t last = inputShapes[i].size() - 1;
                if ((inputShapes[i][last - 1] != 1 && inputShapes[i][last] == 1) ||
                    (inputShapes[i][last - 1] == 1 && inputShapes[i][last] != 1)) {
                    canCollapse = false;
                    break;
                }
            }

            size_t nextJitWorkAmount = currentJitWorkAmount * domain[domain.size() - 2];
            if (fullWorkAmount / nextJitWorkAmount >= minimalConcurrency) {
                currentJitWorkAmount = nextJitWorkAmount;
                // if we cannot use dim collapsing we should use tile2D
                if (!canCollapse) {
                    if (TileRank < maxTileRank) {
                        TileRank++;
                        continue;
                    }

                    break;
                }
                collapsedDims++;
                for (auto &d : inputShapes)
                    collapseLastDims(d, 1);
                for (auto &d : outputShapes)
                    collapseLastDims(d, 1);
                collapseLastDims(domain, 1);
            } else {
                break;
            }
        }
        return collapsedDims > 0;
    };
    return findDimsToCollapse();
}
ov::PartialShape Snippet::canonicalizeBody() {
    auto edgeToBlockedShape = [](const EdgePtr& edge) {
        const auto blockedDesc = edge->getMemory().getDescWithType<BlockedMemoryDesc>();
        std::vector<Dimension> dims;
        // if blockDim == Shape::UNDEFINED_DIM, then it's a dynamic dimension, and we need to recreate a proper dynamic Dim
        for (const auto& d : blockedDesc->getBlockDims())
            dims.emplace_back(d == Shape::UNDEFINED_DIM ? -1 : d);
        ov::PartialShape shape(dims);
        ov::AxisVector blocking(blockedDesc->getOrder());
        ov::element::Type precision = InferenceEngine::details::convertPrecision(blockedDesc->getPrecision());
        return snippets::op::Subgraph::BlockedShape{shape, blocking, precision};
    };
    inputShapeIsBlocked.resize(inputShapes.size(), false);
    masterShapeIsBlocked = false;
    snippets::op::Subgraph::BlockedShapeVector input_blocked_shapes;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto blockedShape = edgeToBlockedShape(getParentEdgesAtPort(i)[0]);
        inputShapeIsBlocked[i] = std::get<0>(blockedShape).size() != std::get<1>(blockedShape).size();
        masterShapeIsBlocked = masterShapeIsBlocked || inputShapeIsBlocked[i];
        input_blocked_shapes.push_back(blockedShape);
    }

    outputShapeIsBlocked.resize(outputShapes.size(), false);
    ov::snippets::op::Subgraph::BlockedShapeVector output_blocked_shapes;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        auto blockedShape = edgeToBlockedShape(getChildEdgesAtPort(i)[0]);
        outputShapeIsBlocked[i] = std::get<0>(blockedShape).size() != std::get<1>(blockedShape).size();
        output_blocked_shapes.push_back(blockedShape);
    }

    const auto& canonicalShape = snippet->canonicalize(output_blocked_shapes, input_blocked_shapes);
    return canonicalShape;
}
void Snippet::createPrimitive() {
    // determine canonicalize, determine master_shape and prepend up to 6D
    // NB! normInputShapes are updated, so body reshape might be needed
    const auto& canonicalShape = canonicalizeBody();
    // initialize by maximum output dimension. Dimensions of outputs should be broadcastable
    tensorRank = std::max(static_cast<size_t>(rank6D), canonicalShape.size());

    const auto config = getSelectedPrimitiveDescriptor()->getConfig();
    auto initDataSizes = [this, config]() {
        const size_t numInputs = inputShapes.size();
        const size_t numOutputs = outputShapes.size();
        dataSize.resize(numInputs + numOutputs);
        for (size_t i = 0; i < numInputs; i++)
            dataSize[i] = config.inConfs[i].getMemDesc()->getPrecision().size();
        for (size_t i = 0; i < numOutputs; i++)
            dataSize[i + numInputs] = config.outConfs[i].getMemDesc()->getPrecision().size();
    };
    initDataSizes();

    jit_snippets_compile_args jcp;
    if (canonicalShape.is_dynamic())
        IE_THROW() << "Snippets: Canonicalization returned dynamic shape in static pipeline";
    masterShape = canonicalShape.get_shape();
    const auto &body = snippet->body_ptr();
    for (const auto& p : body->get_parameters())
        normInputShapes.emplace_back(p->get_output_shape(0));
    for (const auto& r : body->get_results())
        normOutputShapes.emplace_back(r->get_input_shape(0));

    prepareParams();
    jcp.master_shape = masterShape;
    jcp.tile_rank = tileRank;
    generate(&jcp);
    buffer_scratchpad_size = snippet->get_buffer_scratchpad_size();
    buffer_scratchpad.resize(buffer_scratchpad_size * parallel_get_max_threads(), 0);
}

std::vector<VectorDims> Snippet::shapeInfer() {
    // todo: it's very strange that we don't have broadcast_merge_into for cpu shapes
    auto broadcast_merge = [](VectorDims& dst, const VectorDims& src){
        // Ranks are both static.
        auto dst_rank = dst.size();
        auto src_rank = src.size();
        const auto new_rank = std::max(dst_rank, src_rank);
        dst.insert(dst.begin(), new_rank - dst_rank, 1);
        std::vector<Dimension> dims(new_rank);
        bool success = true;
        for (size_t i = 0; i < new_rank; i++) {
            auto dsti = i < (new_rank - dst_rank) ? 1 : dst[i - (new_rank - dst_rank)];
            auto srci = i < (new_rank - src_rank) ? 1 : src[i - (new_rank - src_rank)];
            if (dsti != srci && srci != Shape::UNDEFINED_DIM) {
                if (dsti == 1 || dsti == Shape::UNDEFINED_DIM) {
                    dsti = srci;
                } else {
                    success = false;
                }
            }
        }
        return success;
    };
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        VectorDims inDims {getParentEdgesAtPort(i)[0]->getMemory().getShape().getDims()};
        if (masterShapeIsBlocked && !inputShapeIsBlocked[i])
            inDims.insert(inDims.end(), 1);
        // todo: this is a simple master_shape inference for shape-agnostic operations,
        //  we'll need to account for body operations semantics in the future
        if (i == 0)
            masterShape = inDims;
        else
            broadcast_merge(masterShape, inDims);
        normInputShapes[i] = std::move(inDims);
    }
    if (std::any_of(masterShape.begin(), masterShape.end(), [](const Dim& d){ return d == Shape::UNDEFINED_DIM;})) {
        std::ostringstream errorMessage;
        errorMessage << "Can't compute static master shape for Snippet node with name: " << getName();
        errorMessage << ". Input shapes = ( ";
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            errorMessage << i << " port = " << getParentEdgesAtPort(i)[0]->getMemory().getShape().toString() << ", ";
        }
        errorMessage << "). Master shape = ( " << Shape(masterShape).toString() << " )";
        IE_THROW() << errorMessage.str();
    }

    if (normOutputShapes.size() == 1) {
        normOutputShapes[0] = masterShape;
        return {masterShape};
    }
    std::vector<VectorDims> outputDims;
    std::vector<ov::Shape> new_shapes;
    for (const auto& s : normInputShapes)
        new_shapes.emplace_back(s);
    const auto& outputShapes = snippet->reshape_body(new_shapes);
    for (size_t i = 0; i < outputShapes.size(); i++)
            normOutputShapes[i] = outputShapes[i];
    return normOutputShapes;
}

void Snippet::prepareParams() {
    masterShape = getNormalizedDimsBySize(masterShape, tensorRank);
    std::vector<size_t> original_input_shape_ranks;
    for (auto& pshape : normInputShapes) {
        original_input_shape_ranks.push_back(pshape.size());
        pshape = getNormalizedDimsBySize(pshape, tensorRank);
    }
    for (auto& pshape : normOutputShapes)
        pshape = getNormalizedDimsBySize(pshape, tensorRank);

    tileRank = 1;
    bool dims_collapsed = false;
    fullWorkAmount = std::accumulate(masterShape.begin(), masterShape.end(), 1, std::multiplies<size_t>());
    if (snippet->has_domain_sensitive_ops()) {
        tileRank = 2;
    } else {
        dims_collapsed = optimizeExecDomain(normInputShapes, normOutputShapes, masterShape, tileRank);
    }
    exec_domain = masterShape;

    auto initStartMemoryOffsets = [this]() {
        const auto config = getSelectedPrimitiveDescriptor()->getConfig();
        const size_t numInputs = inputShapes.size();
        start_offset_in.resize(numInputs);
        srcMemPtrs.resize(numInputs);
        for (size_t i = 0; i < numInputs; i++) {
            const auto memPtr = getParentEdgeAt(i)->getMemoryPtr();
            srcMemPtrs[i] = memPtr;
            start_offset_in[i] =  memPtr->getDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize[i];
        }
        const size_t numOutputs = outputShapes.size();
        start_offset_out.resize(numOutputs);
        dstMemPtrs.resize(numOutputs);
        for (size_t i = 0; i < numOutputs; i++) {
            const auto memPtr = getChildEdgeAt(i)->getMemoryPtr();
            dstMemPtrs[i] = memPtr;
            start_offset_out[i] = memPtr->getDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize[i + numInputs];
        }
    };
    // initialize start offsets to src and dst memory
    // Needs to be done for every set of input shapes sce memory ptrs could've updated
    initStartMemoryOffsets();
    std::vector<size_t> scheduler_work_amounts;
    // rename schedulerWorkAmount to harnessWorkAmount?
    harnessWorkAmount = fullWorkAmount;
    const auto rank = exec_domain.size();
    for (auto i = rank - tileRank; i < rank; i++) {
        auto& dim = exec_domain[i];
        harnessWorkAmount /= dim;
        scheduler_work_amounts.push_back(dim);
        dim = 1;
    }

    if (dims_collapsed) {
        std::vector<ov::Shape> new_shapes;
        for (size_t i = 0; i < normInputShapes.size(); i++) {
            const auto norm_shape = normInputShapes[i];
            size_t ndims_to_skip = norm_shape.size() - original_input_shape_ranks[i];
            new_shapes.emplace_back(norm_shape.begin() + ndims_to_skip, norm_shape.end());
        }
        snippet->reshape_body(new_shapes);
    }

    snippet->set_master_shape(ov::PartialShape(masterShape));
    snippet->set_tile_rank(tileRank);
}

bool Snippet::needPrepareParams() const {
    return inputShapesModified() || !schedule.ptr;
}

bool Snippet::canBeInPlace() const {
    if (isDynamic || getParentEdgesAtPort(0)[0]->getParent()->getType() == Type::Input) {
        return false;
    }
    if (getChildEdges().size() != 1) {
        return false;
    }

    for (auto& parentEdge : getParentEdges()) {
        auto parent = parentEdge.lock()->getParent();
        if (parent->getChildEdges().size() != 1)
            return false;

        // WA to prevent memory corruption caused by inplace feature
        if (parent->getType() == Type::Concatenation) {
            for (auto& parentParentEdge : parent->getParentEdges()) {
                auto parentParent = parentParentEdge.lock()->getParent();
                if (parentParent->getChildEdges().size() != 1)
                    return false;
            }
        }
    }
    return getInputShapeAtPort(0) == getOutputShapeAtPort(0);
}

bool Snippet::created() const {
    return getType() == Type::Subgraph;
}

void Snippet::generate(const jit_snippets_compile_args* jcp) {
    ov::pass::Manager pre_dialect;
    pre_dialect.register_pass<ConvertToSwishCPU>();
    if (context->getConfig().inferencePrecision == ov::element::bf16 && snippet->has_domain_sensitive_ops()) {
        // enforce BF16 precisions to supported operations
        // MatMul has to be decomposed to Brgemm operations before enforcement
        // Note, MatMul decomposition will be ran later again for case if BF16 enforcement is not happened
        CPU_REGISTER_PASS_X64(pre_dialect, ov::snippets::pass::MatMulToBrgemm);
        CPU_REGISTER_PASS_X64(pre_dialect, pass::EnforcePrecision, element::f32, element::bf16);
    }

    ov::pass::Manager post_dialect;
    CPU_REGISTER_PASS_X64(post_dialect, ov::intel_cpu::pass::BrgemmToBrgemmCPU);
    CPU_REGISTER_PASS_X64(post_dialect, ov::intel_cpu::pass::SetBrgemmCPUBlockingParams);

    ov::pass::Manager post_precision;
    CPU_REGISTER_PASS_X64(post_precision, ov::intel_cpu::pass::RemoveConverts);
    CPU_REGISTER_PASS_X64(post_precision, ov::intel_cpu::pass::MulAddToFMA);

    ov::snippets::lowered::pass::PassPipeline control_flow_markup_pipeline;
    CPU_REGISTER_PASS_X64(control_flow_markup_pipeline, ov::intel_cpu::pass::BrgemmBlocking);

    ov::snippets::lowered::pass::PassPipeline control_flow_pipeline;
    CPU_REGISTER_PASS_X64(control_flow_pipeline, ov::intel_cpu::pass::FuseLoadStoreConvert);

    schedule = snippet->generate(
        pre_dialect,
        post_dialect,
        post_precision,
        control_flow_markup_pipeline,
        control_flow_pipeline,
        reinterpret_cast<const void*>(jcp));
}

void Snippet::update_ptrs(jit_snippets_call_args& call_args) {
    for (size_t i = 0; i < srcMemPtrs.size(); i++)
        call_args.src_ptrs[i] = reinterpret_cast<const uint8_t*>(srcMemPtrs[i]->getData()) + start_offset_in[i];

    for (size_t i = 0; i < dstMemPtrs.size(); i++)
        call_args.dst_ptrs[i] = reinterpret_cast<uint8_t*>(dstMemPtrs[i]->getData()) + start_offset_out[i];

    if (buffer_scratchpad_size > 0) {
        call_args.buffer_scratchpad_ptr =
                reinterpret_cast<uint8_t*>(buffer_scratchpad.data()) + parallel_get_thread_num() * buffer_scratchpad_size;
    }
}

void Snippet::execute(dnnl::stream strm) {
    if (schedule.ptr == nullptr) {
        IE_THROW() << "Snippet can't use Optimized implementation and can't fallback to reference";
    }
    if (tensorRank == rank6D) {
        schedule_6d();
    } else {
        schedule_nt();
    }
}

void Snippet::schedule_6d() {
    const auto& dom = exec_domain;
    // < N, C, H, W > < 1, 1, N, C*H*W>
    parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
        [&](int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
            int64_t indexes[] = {d0, d1, d2, d3, d4};
            jit_snippets_call_args call_args;
            update_ptrs(call_args);

            schedule.get_callable<kernel>()(indexes, &call_args);
        });
}

void Snippet::schedule_nt() {
    const auto& work_size = exec_domain;
    parallel_nt(0, [&](const int ithr, const int nthr) {
        jit_snippets_call_args call_args;
        update_ptrs(call_args);

        size_t start = 0, end = 0;
        splitter(harnessWorkAmount, nthr, ithr, start, end);

        std::vector<int64_t> indexes(work_size.size() - 1, 0);
        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = work_size.size() - 2; j >= 0; j--) {
                indexes[j] = tmp % work_size[j];
                tmp /= work_size[j];
            }

            schedule.get_callable<kernel>()(indexes.data(), &call_args);
        }
    });
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
