// Copyright (C) 2018-2022 Intel Corporation
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

#include <shared_mutex>

#include <snippets/op/subgraph.hpp>
#include "emitters/cpu_generator.hpp"
#include "snippets_transformations/fuse_load_store_and_convert.hpp"
#include "ngraph_transformations/convert_to_swish_cpu.hpp"

using namespace InferenceEngine;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {


Snippet::Snippet(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : Node(op, eng, cache) {
    host_isa = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ?
        dnnl::impl::cpu::x64::avx512_core : dnnl::impl::cpu::x64::avx2;
    original_snippet = ov::as_type_ptr<ngraph::snippets::op::Subgraph>(op);
    if (!original_snippet) {
        IE_THROW(NotImplemented) << "Node is not an instance of snippets::op::Subgraph";
    }
    isa_num_lanes =  original_snippet->get_generator()->get_target_machine()->get_lanes();
}

void Snippet::copy_snippet() {
    ngraph::OutputVector subgraph_node_inputs;
    for (const auto &input : original_snippet->input_values()) {
        auto new_input = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
        subgraph_node_inputs.push_back(new_input);
    }
    std::shared_ptr<ov::Model> new_body = nullptr;
    // Ticket[79554]: TypeRelaxed ops aren't thread safe so we use mutex to avoid collision in throughput mode
    if (original_snippet->has_type_relaxed_ops()) {
        if (!sharedMutex) {
            IE_THROW() << "Subgraph doesn't have shared mutex";
        }
        std::lock_guard<std::mutex> lock(*sharedMutex.get());
        new_body = ov::clone_model(*original_snippet->get_body().get());
    } else {
        new_body = ov::clone_model(*original_snippet->get_body().get());
    }
    snippet = std::make_shared<ngraph::snippets::op::Subgraph>(subgraph_node_inputs, new_body);
    ngraph::copy_runtime_info(original_snippet, snippet);
    snippet->set_friendly_name(original_snippet->get_friendly_name());
    snippet->set_generator(std::make_shared<CPUGenerator>(host_isa));
}

ov::PartialShape Snippet::prependWithOnes(const PartialShape& dims, size_t rank) {
    if (rank <= dims.size())
        return dims;
    std::vector<ov::Dimension> result(rank, 1);
    std::copy(dims.begin(), dims.end(), &result[rank - dims.size()]);
    return PartialShape {result};
}

VectorDims Snippet::prependWithOnes(const VectorDims& dims, size_t rank) {
    if (rank <= dims.size())
        return dims;
    VectorDims result(rank, 1);
    std::copy(dims.begin(), dims.end(), &result[rank - dims.size()]);
    return result;
}

void Snippet::initSupportedPrimitiveDescriptors() {
    copy_snippet();
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::set<Precision> supportedPrecisions = { Precision::FP32, Precision::I32, Precision::BF16, Precision::I8, Precision::U8 };

    bool dimRanksAreEqual = true;
    for (size_t i = 0; dimRanksAreEqual && i < inputShapes.size(); i++) {
        for (size_t j = 0; dimRanksAreEqual && j < outputShapes.size(); j++) {
            if (inputShapes[i].getRank() != outputShapes[j].getRank())
                dimRanksAreEqual = false;
        }
    }

    const size_t ndims = outputShapes[0].getRank();
    const bool isChannelsFirstApplicable = dnnl::impl::utils::one_of(ndims, 1, 2, 4, 5) && dimRanksAreEqual;
    // Todo: Snippets currently don't support per-channel broadcasting of Blocked descriptors because
    //  canonicalization can't distinguish between <N, C, H, W, c> and <N, C, D, H, W> cases.
    //  See snippets::op::Subgraph::canonicalize for details.
    bool isBlockedApplicable = dnnl::impl::utils::one_of(ndims,  4, 5) && dimRanksAreEqual;

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
        config.dynBatchSupport = false;
        config.inConfs.resize(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); i++) {
            auto precision = getOriginalInputPrecisionAtPort(i);
            if (supportedPrecisions.count(precision) == 0)
                IE_THROW() << "Subgraph node with name `" << getName() << "` doesn't support " << precision << " precision.";

            const auto equalPrecisions = getOriginalOutputPrecisions().size() == 1 &&
                    precision == getOriginalOutputPrecisionAtPort(0);

            BlockedMemoryDesc::CmpMask inputMask = BLOCKED_DESC_SKIP_OFFSET_MASK;
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

            BlockedMemoryDesc::CmpMask outputMask = BLOCKED_DESC_SKIP_OFFSET_MASK;
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
    selectPreferPrimitiveDescriptor(getPrimitivesPriority(), true);
}
InferenceEngine::Precision Snippet::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated && !parentEdge->getParent()->isConstant()) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

void Snippet::calcJITParams(std::vector<int64_t>& offsets, std::vector<int64_t>& sch_offsets, std::vector<bool>& bmask,
                            std::vector<int64_t>& vector_tile_inc, std::vector<int64_t>& scalar_tile_inc) const {
    const size_t numInputs = normInputShapes.size();
    const size_t numParams = numInputs + normOutputShapes.size();

    int io_index = 0;
    bmask.resize(normInputShapes.size() + normOutputShapes.size());
    for (const auto& s : normInputShapes)
        // bmask is true if the input/output is broadcasted
        bmask[io_index++] = masterShape.back() != 1 && s.back() == 1;
    for (const auto& s : normOutputShapes)
        bmask[io_index++] = masterShape.back() != 1 && s.back() == 1;

    // explicit tile increments are needed only for dynamic case
    if (isDynamic) {
        vector_tile_inc.resize(io_index);
        scalar_tile_inc.resize(io_index);
        for (int i = 0; i < io_index; i++) {
            const bool not_broadcasted = !bmask[i];
            vector_tile_inc[i] = isa_num_lanes * dataSize[i] * not_broadcasted;
            scalar_tile_inc[i] = dataSize[i] * not_broadcasted;
        }
    }

    // Note that wen don't need offset for the last dim, since it's handled directly by Load/Store emitters
    const size_t offset_rank = masterShape.size() - 1;
    offsets.resize(numParams * (offset_rank), 1);
    auto offset_calculation = [offset_rank, this](int64_t *off, const std::vector<size_t>& dims, const size_t data_size) {
        size_t k = dims.back();
        for (int i = offset_rank - 1; i >= 0; i--) {
            auto tmp = (dims[i] == masterShape[i]) ? k : 0;
            off[i] = tmp * data_size;
            k *= dims[i];
        }
    };
    for (size_t i = 0; i < numParams; i++) {
        offset_calculation(offsets.data() + i * offset_rank,
                           i < numInputs ? normInputShapes[i] : normOutputShapes[i - numInputs],
                           dataSize[i]);
    }
    // zero-out offsets that wouldn't be applied anyway, see  "if (jcp.master_shape[k] != 1 && offsets[k] != 0)"
    // in TileSchedulerEmitter
    for (size_t i = 0; i < offset_rank; i++) {
        if (masterShape[i] == 1) {
            for (size_t j = i; j < numParams * offset_rank; j += offset_rank)
                offsets[j] = 0;
        }
    }

    sch_offsets = std::vector<int64_t>(numParams, 0);
    if (tileRank > 1) {
        // todo: simplify pointer increment logics. Currently some increments are performed by emitters
        //  (not always, but on condition), and some - by TileScheduler.
        // update offsets for tile 2D because loaders have ptr shifts in some cases and stores have always ptrs shifts
        for (size_t i = 0; i < numParams; i++) {
            // the last offset is ignored, so offsets[offset_rank - 1] is actually outer tile offset
            int64_t off = offsets[(i + 1) * offset_rank - 1];
            const auto& io_shape = i < numInputs ? normInputShapes[i] : normOutputShapes[i - numInputs];
            // off == dataSize[i] only if
            // io_shape[io_shape.size() - 1] == 1 and io_shape[io_shape.size() - 2] == masterShape[io_shape.size() - 2]
            // So the upper Tile is not broadcasted, and the lower Tile is brodcasted if bmask[i] == True.
            // If the lower Tile is broadcasted it doesn't increment data pointer, so we have to shift it in the upper Tile
            // If the lower Tile is NOT broadcasted it shifts data pointers in a normal way
            if (off == dataSize[i] && bmask[i]) {
                sch_offsets[i] = dataSize[i];
            // if outer tile is broadcasted then we need to step back to read the same data once again
            // However, if the lower Tile is also broadcasted, then no step back is needed, since
            // the lower Tile doesn't increment data pointer in this case.
            } else if (io_shape[io_shape.size() - 2] != masterShape[masterShape.size() - 2]
                       && !bmask[i]) {
                sch_offsets[i] = -1 * io_shape.back() * dataSize[i];
            }
        }
    }
}
void Snippet::optimizeExecDomain(std::vector<VectorDims>& inputShapes, std::vector<VectorDims>& outputShapes,
                                 VectorDims &domain, size_t& TileRank) const {
    const size_t minimalConcurrency = parallel_get_max_threads();
    const size_t minimalJitWorkAmount = 256;
    const size_t ds = domain.size();
    if ( ds <= 2 || // not enough dimensions to collapse
         domain[ds-1] >= minimalJitWorkAmount || // There is enough work for 1D Tiles, no need to collapse
         domain[ds-1] * domain[ds-2] >= fullWorkAmount / minimalConcurrency) // There won't be enough work for every thread (even one iter) if we collapse
        return;
    auto findDimsToCollapse = [&]() {
        auto collapseLastDims = [](VectorDims& dims, size_t dimsToCollapse) {
            if (dimsToCollapse >= dims.size() - 1)
                IE_THROW() << "Got invalid number of dims to collapse. Expected < " << dims.size() - 1 << " got " << dimsToCollapse;
            for (int i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
                dims[dims.size() - 1] *= dims[i];
            }

            for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
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
        return domain;
    };
    findDimsToCollapse();
}
ov::PartialShape Snippet::canonicalizeBody() {
    auto edgeToBlockedShape = [](const EdgePtr& edge) {
        const auto blockedDesc = edge->getMemory().GetDescWithType<BlockedMemoryDesc>();
        std::vector<Dimension> dims;
        // if blockDim == Shape::UNDEFINED_DIM, then it's a dynamic dimension, and we need to recreate a proper dynamic Dim
        for (const auto& d : blockedDesc->getBlockDims())
            dims.emplace_back(d == Shape::UNDEFINED_DIM ? 0 : d, d);
        ngraph::PartialShape shape(dims);
        ngraph::AxisVector blocking(blockedDesc->getOrder());
        ngraph::element::Type precision = InferenceEngine::details::convertPrecision(blockedDesc->getPrecision());
        return ngraph::snippets::op::Subgraph::BlockedShape{shape, blocking, precision};
    };
    inputShapeIsBlocked.resize(inputShapes.size(), false);
    masterShapeIsBlocked = false;
    ngraph::snippets::op::Subgraph::BlockedShapeVector input_blocked_shapes;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto blockedShape = edgeToBlockedShape(getParentEdgesAtPort(i)[0]);
        inputShapeIsBlocked[i] = std::get<0>(blockedShape).size() != std::get<1>(blockedShape).size();
        masterShapeIsBlocked = masterShapeIsBlocked || inputShapeIsBlocked[i];
        input_blocked_shapes.push_back(blockedShape);
    }

    outputShapeIsBlocked.resize(outputShapes.size(), false);
    ngraph::snippets::op::Subgraph::BlockedShapeVector output_blocked_shapes;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        auto blockedShape = edgeToBlockedShape(getChildEdgesAtPort(i)[0]);
        outputShapeIsBlocked[i] = std::get<0>(blockedShape).size() != std::get<1>(blockedShape).size();
        output_blocked_shapes.push_back(blockedShape);
    }

    const auto canonicalShape = snippet->canonicalize(output_blocked_shapes, input_blocked_shapes);
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
    jcp.is_static = !isDynamic;
    if (!isDynamic) {
        if (canonicalShape.is_dynamic())
            IE_THROW() << "Snippets: Canonicalization returned dynamic shape in static pipeline";
        masterShape = canonicalShape.get_shape();
        const auto &body = snippet->get_body();
        for (const auto& p : body->get_parameters())
            normInputShapes.emplace_back(p->get_output_shape(0));
        for (const auto& r : body->get_results())
            normOutputShapes.emplace_back(r->get_input_shape(0));

        prepareParams();
        jcp.master_shape = masterShape;
        std::copy(data_offsets.begin(), data_offsets.end(), jcp.data_offsets);
        std::copy(scheduler_offsets.begin(), scheduler_offsets.end(), jcp.scheduler_offsets);
        std::copy(scheduler_work_amounts.begin(), scheduler_work_amounts.end(), jcp.scheduler_work_amounts);
    } else {
        normInputShapes.resize(snippet->get_body()->get_parameters().size());
        normOutputShapes.resize(snippet->get_body()->get_results().size());
        // master_shape size can't change after canonicalization, use this info during dynamic kernel generation
        jcp.master_shape = VectorDims(tensorRank, 0);
    }
    schedule = snippet->generate(reinterpret_cast<const void*>(&jcp));
}

std::vector<VectorDims> Snippet::shapeInfer() const {
    // todo: it's very strange that we don't have broadcast_merge_into for cpu shapes
    auto broadcast_merge = [](VectorDims& dst, const VectorDims& src){
        // Ranks are both static.
        auto dst_rank = dst.size();
        auto src_rank = src.size();
        const auto new_rank = std::max(dst_rank, src_rank);
        dst.insert(dst.begin(), new_rank - dst_rank, 1);
        std::vector<Dimension> dims(new_rank);
        bool success = true;
        for (int64_t i = 0; i < new_rank; i++) {
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
        VectorDims inDims {getParentEdgesAtPort(i)[0]->getMemory().GetShape().getDims()};
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
            errorMessage << i << " port = " << getParentEdgesAtPort(i)[0]->getMemory().GetShape().toString() << ", ";
        }
        errorMessage << "). Master shape = ( " << masterShape << " )";
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
    // here must be all the stuff that could only be done for static shapes, e.g. offset calculation
    // Here it must be all the stuff that could be done once for both static and dynamic shapes

    masterShape = prependWithOnes(masterShape, tensorRank);
    for (auto& pshape : normInputShapes)
        pshape = prependWithOnes(pshape, tensorRank);
    for (auto& pshape : normOutputShapes)
        pshape = prependWithOnes(pshape, tensorRank);

    tileRank = 1;
    fullWorkAmount = std::accumulate(masterShape.begin(), masterShape.end(), 1, std::multiplies<size_t>());
    // optimizeExecDomain will collapse shape dimensions and adjust tile Rank
    optimizeExecDomain(normInputShapes, normOutputShapes, masterShape, tileRank);
    exec_domain = masterShape;

    // todo: probably better to pass a call_args instance
    calcJITParams(data_offsets, scheduler_offsets, broadcasting_mask, vector_tile_increments, scalar_tile_increments);
    auto initStartMemoryOffsets = [this]() {
        const auto config = getSelectedPrimitiveDescriptor()->getConfig();
        const size_t numInputs = inputShapes.size();
        start_offset_in.resize(numInputs);
        srcMemPtrs.resize(numInputs);
        for (size_t i = 0; i < numInputs; i++) {
            const auto memPtr = getParentEdgeAt(i)->getMemoryPtr();
            srcMemPtrs[i] = memPtr;
            start_offset_in[i] =  memPtr->GetDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize[i];
        }
        const size_t numOutputs = outputShapes.size();
        start_offset_out.resize(numOutputs);
        dstMemPtrs.resize(numOutputs);
        for (size_t i = 0; i < numOutputs; i++) {
            const auto memPtr = getChildEdgeAt(i)->getMemoryPtr();
            dstMemPtrs[i] = memPtr;
            start_offset_out[i] = memPtr->GetDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize[i + numInputs];
        }
    };
    // initialize start offsets to src and dst memory
    // Needs to be done for every set of input shapes sce memory ptrs could've updated
    initStartMemoryOffsets();
    scheduler_work_amounts = std::vector<size_t>(maxTileRank, 1);
    // rename schedulerWorkAmount to harnessWorkAmount?
    harnessWorkAmount = fullWorkAmount;
    auto scheduler_it = scheduler_work_amounts.rbegin();
    auto exec_it = exec_domain.rbegin();
    for (int i = 0; i < tileRank; i++, exec_it++, scheduler_it++) {
        harnessWorkAmount /= *exec_it;
        *scheduler_it = *exec_it;
        *exec_it = 1;
    }
}

bool Snippet::needPrepareParams() const {
    return inputShapesModified() || !schedule.ptr;
}

void Snippet::execute(dnnl::stream strm) {
    if (schedule.ptr == nullptr || !canUseOptimizedImpl) {
        IE_THROW() << "Snippet can't use Optimized implementation and can't fallback to reference";
    }
    jit_snippets_call_args call_args;
    for (size_t i = 0; i < srcMemPtrs.size(); i++)
        call_args.src_ptrs[i] = reinterpret_cast<const uint8_t*>(srcMemPtrs[i]->GetData()) + start_offset_in[i];

    for (size_t i = 0; i < dstMemPtrs.size(); i++)
        call_args.dst_ptrs[i] = reinterpret_cast<uint8_t*>(dstMemPtrs[i]->GetData()) + start_offset_out[i];

    if (isDynamic) {
        std::copy(scheduler_offsets.begin(), scheduler_offsets.end(), call_args.scheduler_offsets);
        std::copy(data_offsets.begin(), data_offsets.end(), call_args.data_offsets);
        std::copy(scheduler_work_amounts.begin(), scheduler_work_amounts.end(), call_args.scheduler_work_amounts);
        std::copy(vector_tile_increments.begin(), vector_tile_increments.end(), call_args.vector_tile_increments);
        std::copy(scalar_tile_increments.begin(), scalar_tile_increments.end(), call_args.scalar_tile_increments);
        std::copy(broadcasting_mask.begin(), broadcasting_mask.end(), call_args.broadcasting_mask);
        // scratchpad memory has to ba allocated only once
        // todo: adjust this memory allocation for different supported precisions in future
       if (scratchpad_memory_chunk.empty())
           scratchpad_memory_chunk.resize(parallel_get_num_threads() * isa_num_lanes * inputShapes.size());
       call_args.broadcasting_scratchpad = scratchpad_memory_chunk.data();
       if (tensorRank != rank6D)
           IE_THROW() << "Snippets currently support only up to 6D dynamic inputs";
       // schedule_6d_dynamic is needed only if an input needs to be broadcasted
       // => per-thread broadcasting scratchpads are needed.
       // Fall back to  schedule_6d to avoid scratchpad handling overheads
       if (std::any_of(broadcasting_mask.begin(), broadcasting_mask.end(), [](bool x){return x;}))
           schedule_6d_dynamic(call_args);
       else
           schedule_6d(call_args);
    } else {
        if (tensorRank == rank6D) {
            schedule_6d(call_args);
        } else {
            schedule_nt(call_args);
        }
    }
}

void Snippet::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
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
    size_t harness_num_dims = exec_domain.size() - tileRank;
    if (harness_num_dims > SNIPPETS_MAX_HARNESS_DIMS) {
        canUseOptimizedImpl = false;
        harness_num_dims = SNIPPETS_MAX_HARNESS_DIMS;
    }

    ov::pass::Manager optManager;
    optManager.register_pass<ov::intel_cpu::pass::FuseLoadConvert>();
    optManager.register_pass<ov::intel_cpu::pass::FuseStoreConvert>();
    optManager.register_pass<ConvertToSwishCPU>();

    // LoadConvert uses Load emitter that support conversion from any type to only f32
    optManager.get_pass_config()->set_callback<ov::intel_cpu::pass::FuseLoadConvert>(
            [](const std::shared_ptr<const ov::Node>& n) -> bool {
                if (const auto& convert = std::dynamic_pointer_cast<const ov::op::v0::Convert>(n))
                    return convert->get_destination_type() != ov::element::f32;
                return true;
            });

    // StoreConvert uses Store emitter that support conversion from only f32 to any types
    optManager.get_pass_config()->set_callback<ov::intel_cpu::pass::FuseStoreConvert>(
            [](const std::shared_ptr<const ov::Node>& n) -> bool {
                if (const auto& convert = std::dynamic_pointer_cast<const ov::op::v0::Convert>(n))
                    return convert->get_input_element_type(0) != ov::element::f32;
                return true;
            });

    schedule = snippet->generate(optManager, reinterpret_cast<void*>(&jcp));
}

void Snippet::schedule_6d_dynamic(const jit_snippets_call_args& call_args) const {
    const auto& dom = exec_domain;
    std::vector<jit_snippets_call_args> per_thread_call_args(parallel_get_num_threads(), call_args);
    const size_t scratchpad_size = isa_num_lanes * inputShapes.size();
    // init unique scratchpad per every thread to perform physical broadcasting
    for (int i = 0; i < per_thread_call_args.size(); i++)
        per_thread_call_args[i].broadcasting_scratchpad += i * scratchpad_size;

    parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
                   [&](int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
                       int64_t indexes[] = {d0, d1, d2, d3, d4};
                       schedule.get_callable<kernel>()(indexes, &per_thread_call_args[parallel_get_thread_num()]);
                   });
}

void Snippet::schedule_6d(const jit_snippets_call_args& call_args) const {
    const auto& dom = exec_domain;
    // < N, C, H, W > < 1, 1, N, C*H*W>
    parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
        [&](int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
            int64_t indexes[] = {d0, d1, d2, d3, d4};
            schedule.get_callable<kernel>()(indexes, &call_args);
        });
}

void Snippet::schedule_nt(const jit_snippets_call_args& call_args) const {
    const auto& work_size = exec_domain;
    parallel_nt(0, [&](const int ithr, const int nthr) {
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
