// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph.h"

#include <ie_parallel.hpp>

#include <vector>
#include <algorithm>
#include <array>
#include <tuple>

#include <mkldnn.hpp>
#include <mkldnn_debug.h>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/rt_info.hpp>
#include <ie_ngraph_utils.hpp>

#include <snippets/op/subgraph.hpp>
#include "emitters/cpu_generator.hpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::cpu::x64;
using namespace Xbyak;

MKLDNNSnippetNode::MKLDNNSnippetNode(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    host_isa = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_common) ?
        dnnl::impl::cpu::x64::avx512_common : dnnl::impl::cpu::x64::avx2;

    // Create a deep local copy of the input snippet to perform canonicalization & code generation
    // Todo: Probably better to implement a proper copy constructor
    if (const auto tmp_snippet =  ov::as_type_ptr<ngraph::snippets::op::Subgraph>(op)) {
        ngraph::OutputVector subgraph_node_inputs;
        for (const auto &input : tmp_snippet->input_values()) {
            auto new_input = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
            subgraph_node_inputs.push_back(new_input);
        }
        auto new_body = ov::clone_model(*tmp_snippet->get_body().get());
        snippet = std::make_shared<ngraph::snippets::op::Subgraph>(subgraph_node_inputs, new_body);
        ngraph::copy_runtime_info(tmp_snippet, snippet);
        snippet->set_friendly_name(tmp_snippet->get_friendly_name());
        snippet->set_generator(std::make_shared<CPUGenerator>(host_isa));
    } else {
        IE_THROW(NotImplemented) << "Node is not an instance of snippets::op::Subgraph";
    }
}

void MKLDNNSnippetNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto hasBroadcastByC = [this]() -> bool {
        for (auto op : ngraph::as_type_ptr<ngraph::snippets::op::Subgraph>(snippet)->get_body()->get_ops()) {
            if (ngraph::op::supports_auto_broadcast(op)) {
                auto shape = op->get_input_shape(0);
                // Filter out scalar empty shape Shape{}
                if (ngraph::shape_size(shape) != 1) {
                    for (const auto& input : op->inputs()) {
                        if (input.get_shape().size() > 1 && shape[1] != input.get_shape()[1] && ngraph::shape_size(input.get_shape()) != 1) {
                            return true;
                        }
                    }
                } else {
                    return false;
                }
            }
        }
        return false;
    };

    const Precision supportedPrecision = Precision::FP32;

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
    //  canonicalization can't distinguish between <N, C, H, W, c> and <N, C, D, H, W> cases. So we need to pass an
    //  additional parameter to canonicalization, see snippets::op::Subgraph::canonicalize for details.
    const bool isBlockedApplicable = dnnl::impl::utils::one_of(ndims,  4, 5) && dimRanksAreEqual && !hasBroadcastByC();
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
                size_t blockSize = mayiuse(dnnl::impl::cpu::x64::avx512_common) ? 16 : 8;

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

        size_t offset = std::numeric_limits<size_t>::max();
        NodeConfig config;
        config.dynBatchSupport = false;
        config.inConfs.resize(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); i++) {
            PortConfig portConfig;
            portConfig.inPlace = (!i && canBeInPlace()) ? 0 : -1;
            portConfig.constant = false;
            portConfig.desc = createMemoryDesc(inputShapes[i], supportedPrecision, offset);
            if (inputShapes[i].getDims()[0] == 1) {
                const auto denseDesc = portConfig.desc->as<BlockedMemoryDesc>();
                auto strides = denseDesc->getStrides();
                strides[0] = Shape::UNDEFINED_DIM;
                portConfig.desc = std::make_shared<CpuBlockedMemoryDesc>(denseDesc->getPrecision(),
                                                                         denseDesc->getShape(),
                                                                         denseDesc->getBlockDims(),
                                                                         denseDesc->getOrder(),
                                                                         denseDesc->getOffsetPadding(),
                                                                         denseDesc->getOffsetPaddingToData(),
                                                                         strides);
            }
            config.inConfs[i] = portConfig;
        }
        config.outConfs.resize(outputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); i++) {
            PortConfig portConfig;
            portConfig.inPlace = -1;
            portConfig.constant = false;
            portConfig.desc = createMemoryDesc(outputShapes[i], supportedPrecision, offset);
            if (outputShapes[i].getDims()[0] == 1) {
                const auto denseDesc = portConfig.desc->as<BlockedMemoryDesc>();
                auto strides = denseDesc->getStrides();
                strides[0] = Shape::UNDEFINED_DIM;
                portConfig.desc = std::make_shared<CpuBlockedMemoryDesc>(denseDesc->getPrecision(),
                                                                         denseDesc->getShape(),
                                                                         denseDesc->getBlockDims(),
                                                                         denseDesc->getOrder(),
                                                                         denseDesc->getOffsetPadding(),
                                                                         denseDesc->getOffsetPaddingToData(),
                                                                         strides);
            }
            config.outConfs[i] = portConfig;
        }

        impl_desc_type impl_type = impl_desc_type::unknown;
        if (mayiuse(x64::avx512_common)) {
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

void MKLDNNSnippetNode::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getPrimitivesPriority(), true);
}

void MKLDNNSnippetNode::createPrimitive() {
    // schedule definition part
    // it defines offsets, strides and sizes for snippet kernel scheduling
    define_schedule();

    // code generation part
    // it might be worth to generate explicitly for scheduler work amount for now,
    // but in future some interface should be defined in order to communicate schedule for a kernel
    // or generate schedule for a kernel.
    // Here kernel is generated for most warying dimension by default.
    generate();
}

void MKLDNNSnippetNode::execute(dnnl::stream strm) {
    if (schedule.ptr == nullptr || !canUseOptimizedImpl) {
        IE_THROW() << "MKLDNNSnippetNode can't use Optimized implementation and can't fallback to reference";
    }
    jit_snippets_call_args call_args;
    for (size_t i = 0; i < srcMemPtrs.size(); i++)
        call_args.src_ptrs[i] = reinterpret_cast<const uint8_t*>(srcMemPtrs[i]->GetData()) + start_offset_in[i];

    for (size_t i = 0; i < dstMemPtrs.size(); i++)
        call_args.dst_ptrs[i] = reinterpret_cast<uint8_t*>(dstMemPtrs[i]->GetData()) + start_offset_out[i];

    if (tensorRank == rank6D) {
        schedule_6d(call_args);
    } else {
        schedule_nt(call_args);
    }
}

bool MKLDNNSnippetNode::created() const {
    return getType() == Subgraph;
}

// internal interface for subgraph execution

static size_t argmax_rank(const std::vector<MKLDNNEdgeWeakPtr> &childEdges) {
    auto getOutBlockedDims = [childEdges](int i) {
            if ( auto childEdge =  childEdges[i].lock() )
                return childEdge->getMemory().GetDescWithType<BlockedMemoryDesc>()->getBlockDims();
            else
                IE_THROW() << "Unable to lock childEdge weak_ptr";
            return VectorDims{};
    };
    auto getOutRank = [getOutBlockedDims](int i) {
        return getOutBlockedDims(i).size();
    };
    size_t max_rank_idx = 0;
    size_t max_rank_val = getOutRank(0);
    for (size_t i = 1; i < childEdges.size(); i++) {
        const auto i_rank_val = getOutRank(i);
        if (max_rank_val < i_rank_val) {
            max_rank_idx = i;
            max_rank_val = i_rank_val;
        } else if (max_rank_val == i_rank_val) {
            const auto max_rank_dims = getOutBlockedDims(max_rank_idx);
            const auto i_dims = getOutBlockedDims(i);
            for (size_t j = 0; j < max_rank_val; j++) {
                if (i_dims[j] > max_rank_dims[j]) {
                    max_rank_idx = i;
                    max_rank_val =  i_rank_val;
                    break;
                }
            }
        }
    }
    return max_rank_idx;
}

static auto offset_calculation(std::vector<int64_t>& offset, const std::vector<int64_t>& dims_in, const std::vector<int64_t>& dims_out) -> void {
    int k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
        k *= dims_in[i];
    }
}

static auto collapseLastDims(std::vector<int64_t>& dims, int dimsToCollapse) -> void {
    for (int i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
        dims[dims.size() - 1] *= dims[i];
    }

    for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
        dims[i] = dims[i - dimsToCollapse];
    }

    for (int i = dimsToCollapse - 1; i >= 0; i--) {
        dims[i] = 1;
    }
}

void MKLDNNSnippetNode::define_schedule() {
    const auto config = getSelectedPrimitiveDescriptor()->getConfig();
    const auto dataSize = config.inConfs[0].desc->getPrecision().size();
    // store to use as an execution domain
    max_rank_out_desc_idx = argmax_rank(getChildEdges());
    const auto outBlockingDesc_maxRank = getChildEdgeAt(max_rank_out_desc_idx)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    // initialize by maximum output dimension. Dimensions of outputs should be broadcastable
    tensorRank = std::max(static_cast<size_t>(rank6D), outBlockingDesc_maxRank->getBlockDims().size());

    auto initDims = [this, config, &outBlockingDesc_maxRank](size_t tensorRank) {
        // assume all input sizes are even
        const size_t inputNum = getParentEdges().size();

        dims_in.resize(inputNum);
        for (size_t i = 0; i < inputNum; i++) {
            dims_in[i].resize(tensorRank, 1);
        }

        const auto outOrder = outBlockingDesc_maxRank->getOrder();
        for (size_t i = 0; i < inputNum; i++) {
            auto inBlockingDesc = getParentEdgeAt(i)->getMemory().GetDescWithType<BlockedMemoryDesc>();
            size_t rank = inBlockingDesc->getBlockDims().size();

            // WA to normalize blocked and planar layouts
            // not actual thought, since [§] doesn't support mixed layouts yet
            auto inOrder = inBlockingDesc->getOrder();
            size_t startOff = outOrder.size() != outBlockingDesc_maxRank->getShape().getRank() &&
                              outOrder.back() != inOrder.back() ? 1 : 0;
            for (size_t j = 0; j < rank; j++) {
                dims_in[i][dims_in[i].size() - 1 - j - startOff] = inBlockingDesc->getBlockDims()[rank - 1 - j];
            }
        }

        // assume all output sizes are even
        const size_t outputNum = config.outConfs.size();

        dims_out.resize(outputNum);
        for (size_t i = 0; i < outputNum; i++) {
            dims_out[i].resize(tensorRank, 1);
        }

        for (size_t i = 0; i < outputNum; i++) {
            auto outBlockingDesc = getChildEdgeAt(i)->getMemory().GetDescWithType<BlockedMemoryDesc>();
            size_t rank = outBlockingDesc->getBlockDims().size();

            for (size_t j = 0; j < rank; j++) {
                dims_out[i][dims_out[i].size() - 1 - j] = outBlockingDesc->getBlockDims()[rank - 1 - j];
            }
        }
    };

    auto initOffsets = [this, config, dataSize](size_t tensorRank) {
        // find max rank input among all outputs
        const size_t inputNum = getParentEdges().size();
        offsets_in.resize(inputNum);
        for (size_t i = 0; i < inputNum; i++) {
            offsets_in[i].resize(tensorRank, 1);
            offset_calculation(offsets_in[i], dims_in[i], dims_out[max_rank_out_desc_idx]);
            for (size_t j = 0; j < tensorRank; j++) {
                offsets_in[i][j] *= dataSize;
            }
        }

        start_offset_in.resize(inputNum);
        srcMemPtrs.resize(inputNum);
        for (size_t i = 0; i < inputNum; i++) {
            const auto memPtr = getParentEdgeAt(i)->getMemoryPtr();
            srcMemPtrs[i] = memPtr;
            start_offset_in[i] =  memPtr->GetDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize;
        }

        const size_t outputNum = config.outConfs.size();
        offsets_out.resize(outputNum);
        for (size_t i = 0; i < outputNum; i++) {
            offsets_out[i].resize(tensorRank, 1);
            offset_calculation(offsets_out[i], dims_out[i], dims_out[max_rank_out_desc_idx]);
            for (size_t j = 0; j < tensorRank; j++) {
                offsets_out[i][j] *= dataSize;
            }
        }

        start_offset_out.resize(outputNum);
        dstMemPtrs.resize(outputNum);
        for (size_t i = 0; i < outputNum; i++) {
            const auto memPtr = getChildEdgeAt(i)->getMemoryPtr();
            dstMemPtrs[i] = memPtr;
            start_offset_out[i] = memPtr->GetDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize;
        }
    };

    auto find_dims_to_collapse = [this, config, &outBlockingDesc_maxRank]() -> int {
        int collapsedDims = 0;
        size_t minimalConcurrency = parallel_get_max_threads();
        size_t minimalJitWorkAmount = 256;
        size_t currentJitWorkAmount = dims_out[max_rank_out_desc_idx].back();
        while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount) {
            if (dims_out[max_rank_out_desc_idx].size() - collapsedDims - 2 < 0)
                break;

            bool canCollapse = true;
            for (size_t i = 0; i < dims_in.size(); i++) {
                if ((dims_in[i][dims_in[i].size() - 2] != 1 && dims_in[i][dims_in[i].size() - 1] == 1) ||
                    (dims_in[i][dims_in[i].size() - 2] == 1 && dims_in[i][dims_in[i].size() - 1] != 1)) {
                    canCollapse = false;
                    break;
                }
            }

            size_t nextJitWorkAmount = currentJitWorkAmount * dims_out[max_rank_out_desc_idx][dims_out[max_rank_out_desc_idx].size() - 2];
            if (fullWorkAmount / nextJitWorkAmount >= minimalConcurrency) {
                currentJitWorkAmount = nextJitWorkAmount;
                // if we cannot use dim collapsing we should use tile2D
                if (!canCollapse) {
                    if (tileRank < maxTileRank) {
                        tileRank++;
                        continue;
                    }

                    break;
                }

                collapsedDims++;
                for (size_t i = 0; i < dims_in.size(); i++) {
                    collapseLastDims(dims_in[i], 1);
                }

                for (size_t i = 0; i < dims_out.size(); i++) {
                    collapseLastDims(dims_out[i], 1);
                }
            } else {
                break;
            }
        }
        return collapsedDims;
    };

    auto initSchedulingInfo = [this, dataSize](const size_t tensorRank) -> void {
        // initialize scheduling information
        sch_offsets_in.resize(offsets_in.size(), 0);
        sch_offsets_out.resize(offsets_out.size(), 0);
        sch_dims.resize(maxTileRank, 1);
        sch_dims[maxTileRank-1] = dims_out[max_rank_out_desc_idx].back();
        schedulerWorkAmount = fullWorkAmount / dims_out[max_rank_out_desc_idx].back();
        if (tileRank > 1) {
            sch_dims[maxTileRank - tileRank] = dims_out[max_rank_out_desc_idx][tensorRank - 2];
            schedulerWorkAmount /= dims_out[max_rank_out_desc_idx][tensorRank - 2];
            dims_out[max_rank_out_desc_idx][tensorRank - 2] = 1;

            // update offsets for tile 2D because loaders have ptr shifts in some cases and stores have always ptrs shifts
            for (size_t i = 0; i < offsets_in.size(); i++) {
                int64_t offset = offsets_in[i][tensorRank - 2];
                if ((offset > dataSize) || (offset == 0 && dims_in[i].back() != 1)) {
                    sch_offsets_in[i] = offset - dims_out[max_rank_out_desc_idx].back() * dataSize;
                } else if (offset == dataSize) {
                    sch_offsets_in[i] = offset;
                }
            }

            for (size_t i = 0; i < offsets_out.size(); i++) {
                int64_t offset = offsets_out[i][tensorRank - 2];
                sch_offsets_out[i] = offset - dims_out[max_rank_out_desc_idx].back() * dataSize;
            }
        }
    };

    initDims(tensorRank);

    fullWorkAmount = 1;
    for (size_t i = 0; i < dims_out[max_rank_out_desc_idx].size(); i++) {
        fullWorkAmount *= dims_out[max_rank_out_desc_idx][i];
    }

    const int collapsedDims = find_dims_to_collapse();
    batchDimIdx = tensorRank - outBlockingDesc_maxRank->getBlockDims().size() + collapsedDims;

    initOffsets(tensorRank);
    initSchedulingInfo(tensorRank);
}

void MKLDNNSnippetNode::generate() {
    std::vector<MKLDNNEdgePtr> input_first_row;
    for (size_t i = 0; i < inputShapes.size(); i++)
        input_first_row.push_back(getParentEdgesAtPort(i)[0]);

    auto edgeToBlockedShape = [](const MKLDNNEdgePtr& edge) -> ngraph::snippets::op::Subgraph::BlockedShape {
        const auto blockedDesc = edge->getMemory().GetDescWithType<BlockedMemoryDesc>();
        ngraph::Shape shape(blockedDesc->getBlockDims());
        ngraph::AxisVector blocking(blockedDesc->getOrder());
        ngraph::element::Type precision = InferenceEngine::details::convertPrecision(blockedDesc->getPrecision());
        return std::make_tuple(shape, blocking, precision);
    };
    ngraph::snippets::op::Subgraph::BlockedShapeVector input_blocked_shapes;
    std::transform(input_first_row.begin(), input_first_row.end(), std::back_inserter(input_blocked_shapes), edgeToBlockedShape);

    std::vector<MKLDNNEdgePtr> output_first_row;
    for (size_t i = 0; i < outputShapes.size(); i++)
        // Can it go with difference shape or precision to different edges? I assume no.
        output_first_row.push_back(getChildEdgesAtPort(i)[0]);

    ngraph::snippets::op::Subgraph::BlockedShapeVector output_blocked_shapes;
    std::transform(output_first_row.begin(), output_first_row.end(), std::back_inserter(output_blocked_shapes), edgeToBlockedShape);
    jit_snippets_compile_args jcp;
    jcp.output_dims = dims_out[max_rank_out_desc_idx];
    std::copy(sch_dims.begin(), sch_dims.end(), jcp.scheduler_dims);
    std::copy(sch_offsets_in.begin(), sch_offsets_in.end(), jcp.scheduler_offsets);
    std::copy(sch_offsets_out.begin(), sch_offsets_out.end(), &jcp.scheduler_offsets[sch_offsets_in.size()]);
    size_t harness_num_dims = jcp.output_dims.size() - 1;
    if (harness_num_dims > SNIPPETS_MAX_HARNESS_DIMS) {
        canUseOptimizedImpl = false;
        harness_num_dims = SNIPPETS_MAX_HARNESS_DIMS;
    }
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto b = offsets_in[i].begin();
        std::copy(b, b + harness_num_dims, &jcp.data_offsets[i * harness_num_dims]);
    }
    for (size_t i = 0; i < outputShapes.size(); i++) {
        auto b = offsets_out[i].begin();
        std::copy(b, b + harness_num_dims, &jcp.data_offsets[(inputShapes.size() + i) * harness_num_dims]);
    }
    schedule = snippet->generate(output_blocked_shapes, input_blocked_shapes, reinterpret_cast<void*>(&jcp));
}

void MKLDNNSnippetNode::schedule_6d(const jit_snippets_call_args& call_args) const {
    const auto& dom = dims_out[max_rank_out_desc_idx];
    // < N, C, H, W > < 1, 1, N, C*H*W>
    parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
        [&](int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
            int64_t indexes[] = {d0, d1, d2, d3, d4};
            schedule.get_callable<kernel>()(indexes, &call_args);
        });
}

void MKLDNNSnippetNode::schedule_nt(const jit_snippets_call_args& call_args) const {
    const auto& work_size = dims_out[max_rank_out_desc_idx];
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(schedulerWorkAmount, nthr, ithr, start, end);

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

REG_MKLDNN_PRIM_FOR(MKLDNNSnippetNode, Subgraph);

