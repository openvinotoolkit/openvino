// Copyright (C) 2018-2020 Intel Corporation
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
    if ((snippet_ref = ngraph::as_type_ptr<ngraph::snippets::op::Subgraph>(op))) {
        ngraph::OutputVector subgraph_node_inputs;
        for (const auto &input : snippet_ref->input_values()) {
            auto new_input = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
            subgraph_node_inputs.push_back(new_input);
        }
        auto new_body = ngraph::clone_function(*snippet_ref->get_body().get());
        snippet = std::make_shared<ngraph::snippets::op::Subgraph>(subgraph_node_inputs, new_body);
        ngraph::copy_runtime_info(snippet_ref, snippet);
        snippet->set_friendly_name(snippet_ref->get_friendly_name());
        snippet->set_generator(std::make_shared<CPUGenerator>(host_isa));
    } else {
        snippet_ref.reset();
        snippet.reset();
    }
}

// It's actually initSupportedDescriptors by meaning
void MKLDNNSnippetNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;
}

void MKLDNNSnippetNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<InferenceEngine::Precision> inputPrecisions;
    for (const auto &i : getOriginalInputPrecisions()) {
        inputPrecisions.push_back(i);
    }

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
                // TODO: need investigate
                // bad accuracy for shape {1, 1, 4, 11}, {2, 5, 1, 1}
                // same for disabled collapse dims
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

        // Todo: inhereted from eltwise node. Why offset is max for non-dynamic nodes?
        //  size_t offset = isDynamicNode() ? 0 : std::numeric_limits<size_t>::max();
        size_t offset = std::numeric_limits<size_t>::max();
        NodeConfig config;
        config.dynBatchSupport = outputShapes[0].getRank() > 1 && inputShapes[0] == outputShapes[0];
//        config.inConfs.resize(getParentEdges().size());
//        for (size_t i = 0; i < getParentEdges().size(); i++) {
        config.inConfs.resize(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); i++) {
            PortConfig portConfig;
            portConfig.inPlace = (!i && canBeInPlace() && inputPrecisions[i] == supportedPrecision) ? 0 : -1;
            portConfig.constant = false;
            portConfig.desc = createMemoryDesc(inputShapes[i], supportedPrecision, offset);
            config.inConfs[i] = portConfig;
        }
        config.outConfs.resize(outputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); i++) {
            PortConfig portConfig;
            portConfig.inPlace = -1;
            portConfig.constant = false;
            portConfig.desc = createMemoryDesc(outputShapes[i], supportedPrecision, offset);
            config.outConfs[i] = portConfig;
        }

        impl_desc_type impl_type;
        if (mayiuse(x64::avx512_common)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (mayiuse(x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        } else {
            // Should never hit here: currently only avx512 and avx2 are supported
            impl_type = impl_desc_type::unknown;
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
    define_shedule();

    // code generation part
    // it might be worth to generate explicitly for scheduler work amount for now,
    // but in future some interface should be defined in order to communicate schedule for a kernel
    // or generate schedule for a kernel.
    // Here kernel is generated for most warying dimension by default.
    generate();
}

void MKLDNNSnippetNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (!isConfigDefined(config)) {
        for (size_t i = 0; i < config.inConfs.size(); i++) {
            config.inConfs[i].desc = getDefinedInputDesc(config, i);
        }

        for (size_t i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].desc = getDefinedOutputDesc(config, i);
        }

        initDescriptor(config);
    } else {
        initDescriptor(config);
    }
}

void MKLDNNSnippetNode::execute(dnnl::stream strm) {
    if (schedule.ptr == nullptr) {
        interpret();
        return;
    }
    std::vector<const uint8_t *> inputs(inputShapes.size(), nullptr);
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto & parents = getParentEdgesAtPort(i);
        auto &mem = parents[0]->getMemory();
        inputs[i] = reinterpret_cast<const uint8_t*>(mem.GetData()) + start_offset_in[i];
    }

    std::vector<uint8_t *> outputs(outputShapes.size(), nullptr);
    for (size_t i = 0; i < outputShapes.size(); i++) {
        auto & child = getChildEdgesAtPort(i);
        auto &mem = child[0]->getMemory();
        outputs[i] = reinterpret_cast<uint8_t*>(mem.GetData()) + start_offset_out[i];
    }

    if (isDynBatchEnabled) {
        if ((tileRank > 1) && batchDimIdx == tensorRank - maxTileRank)
            sch_dims[1] = static_cast<size_t>(batchToProcess());
        else
            dims_out[max_rank_out_desc_idx][batchDimIdx] = static_cast<size_t>(batchToProcess());
    }

    if (tensorRank == rank6D) {
        shedule_6d(outputs, inputs);
        return;
    }

    shedule_nt(outputs, inputs);
}

bool MKLDNNSnippetNode::created() const {
    return getType() == Subgraph;
}

// internal interface for subgraph execution

static size_t argmax_rank(const std::vector<MKLDNNEdgeWeakPtr> &childEdges) {
    auto getOutBlockedDims = [childEdges](int i) {
        return (childEdges[i].lock()->getMemory().GetDescWithType<BlockedMemoryDesc>())->getBlockDims();
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

static auto offset_in_calc(std::vector<int64_t>& offset, const std::vector<int64_t>& dims_in, const std::vector<int64_t>& dims_out) -> void {
    int k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
        k *= dims_in[i];
    }
}

static auto offset_out_calc(std::vector<int64_t>& offset, const std::vector<int64_t>& dims) -> void {
    int k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = k;
        k *= dims[i];
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

void MKLDNNSnippetNode::define_shedule() {
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
        // inputs
        // find max rank input among all outputs
        const size_t inputNum = getParentEdges().size();
        offsets_in.resize(inputNum);
        for (size_t i = 0; i < inputNum; i++) {
            offsets_in[i].resize(tensorRank, 1);
            offset_in_calc(offsets_in[i], dims_in[i], dims_out[max_rank_out_desc_idx]);
            for (size_t j = 0; j < tensorRank; j++) {
                offsets_in[i][j] *= dataSize;
            }
        }

        start_offset_in.resize(inputNum);
        for (size_t i = 0; i < inputNum; i++) {
            const auto desc = getParentEdgeAt(i)->getMemory().GetDescWithType<BlockedMemoryDesc>();
            start_offset_in[i] =  desc->getOffsetPadding() * dataSize;
        }

        // outputs
        const size_t outputNum = config.outConfs.size();
        offsets_out.resize(outputNum);
        for (size_t i = 0; i < outputNum; i++) {
            offsets_out[i].resize(tensorRank, 1);
            //offset_out_calc(offsets_out[i], dims_out[i]);
            //Todo NB! Calc in and out offsets in a similar way for test purposes
            offset_in_calc(offsets_out[i], dims_out[i], dims_out[max_rank_out_desc_idx]);
            for (size_t j = 0; j < tensorRank; j++) {
                offsets_out[i][j] *= dataSize;
            }
        }

        start_offset_out.resize(outputNum);
        for (size_t i = 0; i < outputNum; i++) {
            const auto desc = getChildEdgeAt(i)->getMemory().GetDescWithType<BlockedMemoryDesc>();
            start_offset_out[i] = desc->getOffsetPadding() * dataSize;
        }
    };

    auto find_dims_to_collapse = [this, config, &outBlockingDesc_maxRank]() -> int {
        int collapsedDims = 0;
        size_t minimalConcurrency = parallel_get_max_threads();
        size_t minimalJitWorkAmount = 256;
        size_t currentJitWorkAmount = dims_out[max_rank_out_desc_idx].back();
        while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount &&
               // we shouldn't collapse batch dimension in case dynamic batch is enabled
               (!isDynBatchEnabled || (outBlockingDesc_maxRank->getBlockDims().size() - collapsedDims > 2))) {
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
        sch_dims[0] = dims_out[max_rank_out_desc_idx].back();
        if (tileRank > 1) {
            sch_dims[1] = dims_out[max_rank_out_desc_idx][tensorRank - 2];
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

    isDynBatchEnabled = config.dynBatchSupport;

    const int collapsedDims = find_dims_to_collapse();
    batchDimIdx = tensorRank - outBlockingDesc_maxRank->getBlockDims().size() + collapsedDims;
    schedulerWorkAmount = fullWorkAmount / dims_out[max_rank_out_desc_idx].back();

    initOffsets(tensorRank);
    initSchedulingInfo(tensorRank);
}

void MKLDNNSnippetNode::generate() {
    std::vector<MKLDNNEdgePtr> input_first_row;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto edges = getParentEdgesAtPort(i);
        if (getParentEdgesAtPort(i).size() != 1) {
            IE_THROW() << "Snippet layer " << getName() << " has >= 1 number of parent edges at port " << i;
        }

        input_first_row.push_back(edges[0]);
    }
    ngraph::snippets::op::Subgraph::BlockedShapeVector input_shapes;
    std::transform(input_first_row.begin(), input_first_row.end(), std::back_inserter(input_shapes),
                [](const MKLDNNEdgePtr& edge) -> ngraph::snippets::op::Subgraph::BlockedShape {
        const auto blockedDesc = edge->getMemory().GetDescWithType<BlockedMemoryDesc>();
        ngraph::Shape shape(blockedDesc->getBlockDims());
        ngraph::AxisVector blocking(blockedDesc->getOrder());
        ngraph::element::Type precision = (blockedDesc->getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
        return std::make_tuple(shape, blocking, precision);
    });

    std::vector<MKLDNNEdgePtr> output_first_row;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        auto edges = getChildEdgesAtPort(i);
        // Can it go with difference shape or precision to different edges? I assume no.
        output_first_row.push_back(edges[0]);
    }

    ngraph::snippets::op::Subgraph::BlockedShapeVector output_shapes;
    std::transform(output_first_row.begin(), output_first_row.end(), std::back_inserter(output_shapes),
                [](const MKLDNNEdgePtr& edge) -> ngraph::snippets::op::Subgraph::BlockedShape {
        const auto blockedDesc = edge->getMemory().GetDescWithType<BlockedMemoryDesc>();
        ngraph::Shape shape(blockedDesc->getBlockDims());
        ngraph::AxisVector blocking(blockedDesc->getOrder());
        ngraph::element::Type precision = (blockedDesc->getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
        return std::make_tuple(shape, blocking, precision);
    });

    schedule = snippet->generate(output_shapes, input_shapes);
}

void MKLDNNSnippetNode::shedule_6d(const std::vector<uint8_t *>& outputs, const std::vector<const uint8_t *>& inputs) const {
    size_t n = inputs.size();
    size_t m = outputs.size();
    auto dom = dims_out[max_rank_out_desc_idx];

    // SchduleInfo/ Domen = d0 .. dN
    CallArgs sch;
    for (const auto& d : sch_dims)
        sch.push(d);

    // SchduleInfo/ Offsets
    CallArgs off;
    for (const auto& offset : sch_offsets_in)
        off.push(offset);
    for (const auto& offset : sch_offsets_out)
        off.push(offset);

    // < N, C, H, W > < 1, 1, N, C*H*W>
    parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
        [&](size_t d0, size_t d1, size_t d2, size_t d3, size_t d4) {
            CallArgs ca;
            // Benchmarking for overhead
            for (size_t i = 0; i < n; i++) {
                ca.push(inputs[i] + d0*offsets_in[i][0] + d1*offsets_in[i][1] + d2*offsets_in[i][2] + d3*offsets_in[i][3] + d4*offsets_in[i][4]);
            }
            for (size_t i = 0; i < m; i++) {
                ca.push(outputs[i] + d0*offsets_out[i][0] + d1*offsets_out[i][1] + d2*offsets_out[i][2] + d3*offsets_out[i][3] + d4*offsets_out[i][4]);
            }

            schedule.get_callable<kernel>()(ca.raw(), sch.raw(), off.raw());
        });
}

void MKLDNNSnippetNode::shedule_nt(const std::vector<uint8_t *>& outputs, const std::vector<const uint8_t *>& inputs) const {
    size_t inputNum = inputs.size();
    size_t outputsNum = outputs.size();

    IE_THROW() << "Fix shedule_nt for a new calling convention";

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(schedulerWorkAmount, nthr, ithr, start, end);

        auto work_size = dims_out[max_rank_out_desc_idx];
        std::vector<size_t> counters(work_size.size(), 0);

        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = work_size.size() - 2; j >= 0; j--) {
                counters[j] = tmp % work_size[j];
                tmp /= work_size[j];
            }

            size_t index_in[7] = {0};
            for (int i = 0; i < inputNum; i++) {
                index_in[i] = 0;
                for (int j = 0; j < counters.size(); j++) {
                    index_in[i] += counters[j] * offsets_in[i][j];
                }
            }

            size_t index_out[7] = {0};
            for (int i = 0; i < outputsNum; i++) {
                for (int j = 0; j < counters.size(); j++) {
                    index_out[i] += counters[j] * offsets_out[i][j];
                }
            }

            union param {
                const float* ptr;
                size_t len;
            };

            std::array<param, 8> args;

            for (size_t i = 0; i < inputNum; i++) {
                args[i].ptr = reinterpret_cast<const float*>(inputs[i] + index_in[i]);
            }

            for (size_t i = 0; i < outputsNum; i++) {
                args[inputNum+i].ptr = reinterpret_cast<const float*>(outputs[i] + index_out[i]);
            }

            args[inputNum+outputsNum].len = static_cast<size_t>(work_size[work_size.size() - 1]);

            typedef void (*ker)(const void *);
            ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(schedule.ptr));
            k(&args[0]);
        }
    });
}

void MKLDNNSnippetNode::interpret() const {
    ngraph::HostTensorVector inputs;
    auto params = snippet->get_body()->get_parameters();
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto & parents = getParentEdgesAtPort(i);
        auto &mem = parents[0]->getMemory();
        auto type = snippet->get_input_element_type(i);
        auto &shape = params[i]->get_shape();
        inputs.push_back(std::make_shared<ngraph::HostTensor>(type, shape, mem.GetPtr()));
    }

    ngraph::HostTensorVector outputs;
    auto results = snippet->get_body()->get_results();
    for (size_t i = 0; i < outputShapes.size(); i++) {
        auto & child = getChildEdgesAtPort(i);
        auto &mem = child[0]->getMemory();
        auto type = snippet->get_output_element_type(i);
        auto &shape = results[i]->get_shape();
        outputs.push_back(std::make_shared<ngraph::HostTensor>(type, shape, mem.GetPtr()));
    }
    snippet->evaluate(outputs, inputs);
}

REG_MKLDNN_PRIM_FOR(MKLDNNSnippetNode, Subgraph);

