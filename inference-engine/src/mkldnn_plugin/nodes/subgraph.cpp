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

    enum LayoutType {
        Planar,
        ChannelsFirst,
        Blocked
    };

    auto hasBroadcastByC = [this]() -> bool {
        for (auto op : ngraph::as_type_ptr<ngraph::snippets::op::Subgraph>(snippet)->get_body()->get_ops()) {
            if (ngraph::op::supports_auto_broadcast(op)) {
                auto shape = op->input(0).get_shape();
                // Filter out scalar empty shape Shape{}
                if (ngraph::shape_size(shape) != 1) {
                    for (auto input : op->inputs()) {
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

    auto initDesc = [&] (LayoutType lt) -> PrimitiveDescInfo {
        auto div_up = [](const int a, const int b) -> int {
            if (!b)
                return 0;
            return (a + b - 1) / b;
        };

        auto createMemoryDesc = [lt, div_up, this](MKLDNNEdgePtr edge, Precision prc, size_t offset) -> TensorDesc {
            if (lt == ChannelsFirst && edge->getDims().ndims() != 1) {
                auto dims = edge->getDims().ToSizeVector();
                auto ndims = dims.size();
                std::vector<size_t> order(ndims);
                std::iota(order.begin(), order.end(), 0);
                if (ndims > 1) {
                    order.erase(order.begin() + 1);
                    order.push_back(1);
                }

                std::vector<size_t> blocks(ndims);
                for (size_t i = 0; i < order.size(); i++) {
                    blocks[i] = dims[order[i]];
                }

                return MKLDNNMemoryDesc(TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset}));
            } else if (lt == Blocked && edge->getDims()[1] != 1 && edge->getDims().ndims() != 1) {
                size_t blockSize = host_isa == dnnl::impl::cpu::x64::avx512_common ? 16 : 8;

                std::vector<size_t> blocks = edge->getDims().ToSizeVector();
                std::vector<size_t> order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                blocks[1] = div_up(blocks[1], blockSize);
                blocks.push_back(blockSize);
                order.push_back(1);

                return MKLDNNMemoryDesc(TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset}));
            } else {
                std::vector<size_t> blocks = edge->getDims().ToSizeVector();
                std::vector<size_t> order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                return MKLDNNMemoryDesc(TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset}));
            }
        };

        size_t offset = std::numeric_limits<size_t>::max();
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = getChildEdgeAt(0)->getDims().ndims() > 1 && getChildEdgeAt(0)->getDims() == getParentEdgeAt(0)->getDims();

        for (auto k = 0; k < this->inDims.size(); k++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = (!k && canBeInPlace() && inputPrecisions[k] == Precision(Precision::FP32)) ? 0 : -1;
            dataConfig.constant = false;

            dataConfig.desc = createMemoryDesc(getParentEdgesAtPort(k)[0], /*inputPrecisions[i]*/Precision(Precision::FP32), offset);

            config.inConfs.push_back(dataConfig);
        }

        for (auto k = 0; k < this->outDims.size(); k++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;

            dataConfig.desc = createMemoryDesc(getChildEdgeAt(k), Precision(Precision::FP32), offset);

            config.outConfs.push_back(dataConfig);
        }

        return {config, impl_desc_type::unknown, MKLDNNMemoryDesc(config.outConfs[0].desc).getFormat()};
    };

    bool isChannelsFirstApplicable = dnnl::impl::utils::one_of(getChildEdgeAt(0)->getDims().ndims(), 1, 2, 4, 5);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        isChannelsFirstApplicable = isChannelsFirstApplicable && dnnl::impl::utils::one_of(getParentEdgeAt(i)->getDims().ndims(), 1, 2, 4, 5);

        for (size_t j = 0; j < getChildEdges().size(); j++) {
            isChannelsFirstApplicable = isChannelsFirstApplicable && getChildEdgeAt(j)->getDims().ndims() == getParentEdgeAt(i)->getDims().ndims();
        }
    }

    bool isBlockedApplicable = dnnl::impl::utils::one_of(getChildEdgeAt(0)->getDims().ndims(),  4, 5);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        isBlockedApplicable = isBlockedApplicable && dnnl::impl::utils::one_of(getParentEdgeAt(i)->getDims().ndims(), 4, 5);

        for (size_t j = 0; j < getChildEdges().size(); j++) {
            isBlockedApplicable = isBlockedApplicable && getChildEdgeAt(j)->getDims().ndims() == getParentEdgeAt(i)->getDims().ndims();
        }
    }

    if (isChannelsFirstApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
    if (isBlockedApplicable && !hasBroadcastByC())
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
    // or generate schedule for a kernel..
    // Here kernel is generated for most warying dimension by default.
    generate();
}

void MKLDNNSnippetNode::execute(dnnl::stream strm) {
    if (schedule.ptr == nullptr) {
        interpret();
        return;
    }
    std::vector<const uint8_t *> inputs(inDims.size(), nullptr);
    for (size_t i = 0; i < inDims.size(); i++) {
        auto & parents = getParentEdgesAtPort(i);
        auto &mem = parents[0]->getMemory();
        inputs[i] = reinterpret_cast<const uint8_t*>(mem.GetData()) + start_offset_in[i];
    }

    std::vector<uint8_t *> outputs(outDims.size(), nullptr);
    for (size_t i = 0; i < outDims.size(); i++) {
        auto & child = getChildEdgesAtPort(i);
        auto &mem = child[0]->getMemory();
        outputs[i] = reinterpret_cast<uint8_t*>(mem.GetData()) + start_offset_out[i];
    }

    if (isDynBatchEnabled) {
        if (!isCollapsing && batchDimIdx == tensorRank - maxTileRank)
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

static auto argmax_rank(const std::vector<DataConfig>& conf) -> size_t {
    auto max_rank_out_desc = conf[0].desc.getBlockingDesc().getBlockDims().size();
    size_t max_rank_out_desc_idx = 0;
    auto i = 0;
    for (auto& d : conf) {
        const auto desc_rank = d.desc.getBlockingDesc().getBlockDims().size();
        if (max_rank_out_desc < desc_rank) {
            max_rank_out_desc_idx = i;
            max_rank_out_desc =  desc_rank;
        } else if (max_rank_out_desc == desc_rank) {
            const auto max_rank_dims = conf[max_rank_out_desc_idx].desc.getBlockingDesc().getBlockDims();
            const auto desc_dims = d.desc.getBlockingDesc().getBlockDims();
            for (int j = 0; j < desc_rank; j++) {
                if (desc_dims[j] > max_rank_dims[j]) {
                    max_rank_out_desc_idx = i;
                    max_rank_out_desc =  desc_rank;
                    break;
                }
            }
        }
        i++;
    }
    std::cout << "max_rank_out_desc " << max_rank_out_desc << " max_rank_out_desc_idx " << max_rank_out_desc_idx << std::endl;
    return max_rank_out_desc_idx;
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
    const auto dataSize = config.inConfs[0].desc.getPrecision().size();

    auto initDims = [this, config](size_t tensorRank) {
        // assume all input sizes are even
        size_t inputNum = getParentEdges().size();
        std::cout << "inputNum = " << inputNum << std::endl;

        dims_in.resize(inputNum);
        for (int i = 0; i < inputNum; i++) {
            dims_in[i].resize(tensorRank, 1);
        }

        auto outOrder = config.outConfs[max_rank_out_desc_idx].desc.getBlockingDesc().getOrder();
        for (int i = 0; i < inputNum; i++) {
            size_t rank = config.inConfs[i].desc.getBlockingDesc().getBlockDims().size();

            // WA to normalize blocked and planar layouts
            // not actual thought, since [§] doesn't support mixed layouts yet
            auto inOrder = config.inConfs[i].desc.getBlockingDesc().getOrder();
            size_t startOff = outOrder.size() != config.outConfs[max_rank_out_desc_idx].desc.getDims().size() &&
                              outOrder.back() != inOrder.back() ? 1 : 0;
            std::cout << "startOff = " << startOff << std::endl;
            for (int j = 0; j < rank; j++) {
                dims_in[i][dims_in[i].size() - 1 - j - startOff]
                = config.inConfs[i].desc.getBlockingDesc().getBlockDims()[rank - 1 - j];
            }
        }

        // assume all output sizes are even
        size_t outputNum = config.outConfs.size();//getChildEdges().size();
        std::cout << "outputNum = " << outputNum << std::endl;

        dims_out.resize(outputNum);
        for (int i = 0; i < outputNum; i++) {
            dims_out[i].resize(tensorRank, 1);
        }

        for (int i = 0; i < outputNum; i++) {
            size_t rank = config.outConfs[i].desc.getBlockingDesc().getBlockDims().size();

            for (int j = 0; j < rank; j++) {
                dims_out[i][dims_out[i].size() - 1 - j]
                = config.outConfs[i].desc.getBlockingDesc().getBlockDims()[rank - 1 - j];
            }
        }
    };

    auto initOffsets = [this, config, dataSize](size_t tensorRank) {
        // inputs
        // find max rank input among all outputs
        size_t inputNum = getParentEdges().size();
        offsets_in.resize(inputNum);
        for (int i = 0; i < inputNum; i++) {
            offsets_in[i].resize(tensorRank, 1);
            offset_in_calc(offsets_in[i], dims_in[i], dims_out[max_rank_out_desc_idx]);
            for (int j = 0; j < tensorRank; j++) {
                offsets_in[i][j] *= dataSize;
            }
        }

        start_offset_in.resize(inputNum);
        for (size_t i = 0; i < inputNum; i++) {
            start_offset_in[i] = getParentEdgeAt(i)->getMemory().GetDescriptor().data.offset0 * dataSize;
        }

        // outputs
        size_t outputNum = config.outConfs.size();//getChildEdges().size();
        offsets_out.resize(outputNum);
        for (int i = 0; i < outputNum; i++) {
            offsets_out[i].resize(tensorRank, 1);
            //offset_out_calc(offsets_out[i], dims_out[i]);
            //Todo NB! Calc in and out offsets in a similar way for test purposes
            offset_in_calc(offsets_out[i], dims_out[i], dims_out[max_rank_out_desc_idx]);
            for (int j = 0; j < tensorRank; j++) {
                offsets_out[i][j] *= dataSize;
            }
        }

        start_offset_out.resize(outputNum);
        for (int i = 0; i < outputNum; i++) {
            start_offset_out[i] = getChildEdgeAt(i)->getMemory().GetDescriptor().data.offset0 * dataSize;
        }
    };

    auto find_dims_to_collapse = [this, config]() -> int {
        int collapsedDims = 0;
        // we support dim collapsing only for equal in and out dims without broadcasting otherwise we use tile2D
        for (int i = 0; i < dims_in.size(); i++) {
            for (int j = 0; j < dims_out.size(); j++) {
                if (dims_in[i] != dims_out[j])
                    return collapsedDims;
            }
        }

        size_t minimalConcurrency = parallel_get_max_threads();
        size_t minimalJitWorkAmount = 256;
        size_t currentJitWorkAmount = dims_out[max_rank_out_desc_idx].back();
        bool hasDifferentDims = false;
        isCollapsing = true;
        while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount &&
               // we shouldn't collapse batch dimension in case dynamic batch is enabled
               (!isDynBatchEnabled || (config.outConfs[max_rank_out_desc_idx].desc.getBlockingDesc().getBlockDims().size() - collapsedDims > 2))) {
            if (dims_out[max_rank_out_desc_idx].size() - collapsedDims - 2 < 0)
                break;

            for (int j = 1; j < dims_in.size(); j++) {
                if (dims_in[j].back() != dims_in[0].back()) {
                    hasDifferentDims = true;
                }
            }

            bool canCollapse = true;
            for (int i = 0; i < dims_in.size(); i++) {
                if (dims_in[i][dims_in[i].size() - 2] != 1) {
                    if (dims_in[i][dims_in[i].size() - 1] == 1) {
                        canCollapse = false;
                        break;
                    }

                    if (hasDifferentDims) {
                        canCollapse = false;
                        break;
                    }
                }
            }

            if (!canCollapse) {
                break;
            }

            size_t nextJitWorkAmount = currentJitWorkAmount * dims_out[max_rank_out_desc_idx][dims_out[max_rank_out_desc_idx].size() - 2];
            if (fullWorkAmount / nextJitWorkAmount >= minimalConcurrency) {
                currentJitWorkAmount = nextJitWorkAmount;
                collapsedDims++;

                for (int i = 0; i < dims_in.size(); i++) {
                    collapseLastDims(dims_in[i], 1);
                }

                for (int i = 0; i < dims_out.size(); i++) {
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
        if (!isCollapsing) {
            sch_dims[1] = dims_out[max_rank_out_desc_idx][tensorRank - 2];
            dims_out[max_rank_out_desc_idx][tensorRank - 2] = 1;

            // update offsets for tile 2D because loaders have ptr shifts in some cases and stores have always ptrs shifts
            for (auto i = 0; i < offsets_in.size(); i++) {
                int64_t offset = offsets_in[i][tensorRank - 2];
                if (offset > dataSize || offset == 0 && dims_in[i].back() != 1) {
                    sch_offsets_in[i] = offset - dims_out[max_rank_out_desc_idx].back() * dataSize;
                } else if (offset == dataSize) {
                    sch_offsets_in[i] = offset;
                }
            }

            for (auto i = 0; i < offsets_out.size(); i++) {
                int64_t offset = offsets_out[i][tensorRank - 2];
                sch_offsets_out[i] = offset - dims_out[max_rank_out_desc_idx].back() * dataSize;
            }
        }
    };

    // store to use as an execution domain
    max_rank_out_desc_idx = argmax_rank(config.outConfs);

    // initialize by maximum output dimension. Dimensions of outputs should be broadcastable
    tensorRank = std::max(static_cast<size_t>(rank6D), config.outConfs[max_rank_out_desc_idx].desc.getBlockingDesc().getBlockDims().size());

    initDims(tensorRank);

    fullWorkAmount = 1;
    for (int i = 0; i < dims_out[max_rank_out_desc_idx].size(); i++) {
        fullWorkAmount *= dims_out[max_rank_out_desc_idx][i];
    }

    isDynBatchEnabled = config.dynBatchSupport;

    const int collapsedDims = find_dims_to_collapse();
    batchDimIdx = tensorRank - config.outConfs[max_rank_out_desc_idx].desc.getBlockingDesc().getBlockDims().size() + collapsedDims;
    schedulerWorkAmount = fullWorkAmount / dims_out[max_rank_out_desc_idx].back();

    initOffsets(tensorRank);
    initSchedulingInfo(tensorRank);
}

void MKLDNNSnippetNode::generate() {
    std::vector<MKLDNNEdgePtr> input_first_row;
    for (size_t i = 0; i < inDims.size(); i++) {
        auto edges = getParentEdgesAtPort(i);
        if (getParentEdgesAtPort(i).size() != 1) {
            IE_THROW() << "Snippet layer " << getName() << " has >= 1 number of parent edges at port " << i;
        }

        input_first_row.push_back(edges[0]);
    }
    ngraph::snippets::op::Subgraph::BlockedShapeVector input_shapes;
    std::transform(input_first_row.begin(), input_first_row.end(), std::back_inserter(input_shapes),
                [](const MKLDNNEdgePtr& edge) -> ngraph::snippets::op::Subgraph::BlockedShape {
        ngraph::Shape shape(edge->getDesc().getBlockingDesc().getBlockDims());
        ngraph::AxisVector blocking(edge->getDesc().getBlockingDesc().getOrder());
        ngraph::element::Type precision = (edge->getDesc().getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
        return std::make_tuple(shape, blocking, precision);
    });

    std::vector<MKLDNNEdgePtr> output_first_row;
    for (size_t i = 0; i < outDims.size(); i++) {
        auto edges = getChildEdgesAtPort(i);
        // Can it go with difference shape or precision to different edges? I assume no.
        output_first_row.push_back(edges[0]);
    }

    ngraph::snippets::op::Subgraph::BlockedShapeVector output_shapes;
    std::transform(output_first_row.begin(), output_first_row.end(), std::back_inserter(output_shapes),
                [](const MKLDNNEdgePtr& edge) -> ngraph::snippets::op::Subgraph::BlockedShape {
        ngraph::Shape shape(edge->getDesc().getBlockingDesc().getBlockDims());
        ngraph::AxisVector blocking(edge->getDesc().getBlockingDesc().getOrder());
        ngraph::element::Type precision = (edge->getDesc().getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
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

    // OffsetInfo
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

            schedule.get_callable<kernel>()(ca.raw(), off.raw(), sch.raw());
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

bool MKLDNNSnippetNode::evaluate(const ngraph::HostTensorVector& outputs, const ngraph::HostTensorVector& inputs) const {
    union param {
        float* ptr;
        size_t len;
    };

    std::array<param, 8> args;

    auto work_size = schedule.work_size;
    size_t in_size = inputs.size();
    size_t out_size = outputs.size();

    // FixMe: linearization conflicts with post increment generation logic for now...
    if (false && schedule.is_flat) {
        for (size_t i = 0; i < in_size; i++) {
            args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr());
        }

        for (size_t i = 0; i < out_size; i++) {
            args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr());
        }

        args[in_size+out_size].len = ngraph::shape_size(work_size);

        typedef void (*ker)(const void *);
        ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(schedule.ptr));
        k(&args[0]);
    } else if (work_size.size() <= 4) {
        auto deduce_strides = [](const ngraph::Shape& p, const ngraph::Shape& w) -> std::array<size_t, 4> {
            size_t h = (p[2] != w[2] ? 0 : p[3]);
            size_t c = (p[1] != w[1] ? 0 : p[3]*p[2]);
            size_t n = (p[0] != w[0] ? 0 : p[3]*p[2]*p[1]);
            return std::array<size_t, 4> {1, n, c, h};
        };

        std::vector<std::array<size_t, 4>> in_shapes;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        std::vector<std::array<size_t, 4>> out_shapes;
        std::transform(outputs.begin(), outputs.end(), std::back_inserter(out_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        for (size_t n = 0; n < work_size[0]; n++) {
            for (size_t c = 0; c < work_size[1]; c++) {
                for (size_t h = 0; h < work_size[2]; h++) {
                    // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                    for (size_t i = 0; i < in_size; i++) {
                        auto paramShape = in_shapes[i];
                        args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                            + h*paramShape[3]
                            + c*paramShape[2]
                            + n*paramShape[1];
                    }

                    for (size_t i = 0; i < out_size; i++) {
                        auto paramShape = out_shapes[i];
                        args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                            + h*paramShape[3]
                            + c*paramShape[2]
                            + n*paramShape[1];
                    }

                    args[in_size+out_size].len = work_size[3];

                    typedef void (*ker)(const void *);
                    ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(schedule.ptr));
                    k(&args[0]);
                }
            }
        }
    } else if (work_size.size() == 5) {
        auto deduce_strides = [](const ngraph::Shape& p, const ngraph::Shape& ws) -> std::array<size_t, 5> {
            size_t w = (p[3] != ws[3] ? 0 : p[4]);
            size_t h = (p[2] != ws[2] ? 0 : p[4]*p[3]);
            size_t c = (p[1] != ws[1] ? 0 : p[4]*p[3]*p[2]);
            size_t n = (p[0] != ws[0] ? 0 : p[4]*p[3]*p[2]*p[1]);

            return std::array<size_t, 5> {1, n, c, h, w};
        };

        std::vector<std::array<size_t, 5>> in_shapes;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        std::vector<std::array<size_t, 5>> out_shapes;
        std::transform(outputs.begin(), outputs.end(), std::back_inserter(out_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        for (size_t n = 0; n < work_size[0]; n++) {
            for (size_t c = 0; c < work_size[1]; c++) {
                for (size_t h = 0; h < work_size[2]; h++) {
                    for (size_t w = 0; w < work_size[3]; w++) {
                        // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                        for (size_t i = 0; i < in_size; i++) {
                            auto paramShape = in_shapes[i];
                            args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                                + w*paramShape[4]
                                + h*paramShape[3]
                                + c*paramShape[2]
                                + n*paramShape[1];
                        }

                        for (size_t i = 0; i < out_size; i++) {
                            auto paramShape = out_shapes[i];
                            args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                                + w*paramShape[4]
                                + h*paramShape[3]
                                + c*paramShape[2]
                                + n*paramShape[1];
                        }

                        args[in_size+out_size].len = work_size[4];

                        typedef void (*ker)(const void *);
                        ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(schedule.ptr));
                        k(&args[0]);
                    }
                }
            }
        }
    } else {
        auto deduce_strides = [](const ngraph::Shape& p, const ngraph::Shape& ws) -> std::array<size_t, 6> {
            size_t v = (p[4] != ws[4] ? 0 : p[5]);
            size_t w = (p[3] != ws[3] ? 0 : p[5]*p[4]);
            size_t h = (p[2] != ws[2] ? 0 : p[5]*p[4]*p[3]);
            size_t c = (p[1] != ws[1] ? 0 : p[5]*p[4]*p[3]*p[2]);
            size_t n = (p[0] != ws[0] ? 0 : p[5]*p[4]*p[3]*p[2]*p[1]);

            return std::array<size_t, 6> {1, n, c, h, w, v};
        };

        std::vector<std::array<size_t, 6>> in_shapes;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        std::vector<std::array<size_t, 6>> out_shapes;
        std::transform(outputs.begin(), outputs.end(), std::back_inserter(out_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        for (size_t n = 0; n < work_size[0]; n++) {
            for (size_t c = 0; c < work_size[1]; c++) {
                for (size_t h = 0; h < work_size[2]; h++) {
                    for (size_t w = 0; w < work_size[3]; w++) {
                        for (size_t v = 0; v < work_size[4]; v++) {
                            // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                            for (size_t i = 0; i < in_size; i++) {
                                auto paramShape = in_shapes[i];
                                args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                                    + v*paramShape[5]
                                    + w*paramShape[4]
                                    + h*paramShape[3]
                                    + c*paramShape[2]
                                    + n*paramShape[1];
                            }

                            for (size_t i = 0; i < out_size; i++) {
                                auto paramShape = out_shapes[i];
                                args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                                    + v*paramShape[5]
                                    + w*paramShape[4]
                                    + h*paramShape[3]
                                    + c*paramShape[2]
                                    + n*paramShape[1];
                            }

                            args[in_size+out_size].len = work_size[5];

                            typedef void (*ker)(const void *);
                            ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(schedule.ptr));
                            k(&args[0]);
                        }
                    }
                }
            }
        }
    }

    return true;
}

void MKLDNNSnippetNode::interpret() const {
    ngraph::HostTensorVector inputs;
    auto params = snippet->get_body()->get_parameters();
    for (size_t i = 0; i < inDims.size(); i++) {
        auto & parents = getParentEdgesAtPort(i);
        auto &mem = parents[0]->getMemory();
        auto type = snippet->input(i).get_element_type();
        auto &shape = params[i]->get_shape();
        inputs.push_back(std::make_shared<ngraph::HostTensor>(type, shape, mem.GetPtr()));
    }

    ngraph::HostTensorVector outputs;
    auto results = snippet->get_body()->get_results();
    for (size_t i = 0; i < outDims.size(); i++) {
        auto & child = getChildEdgesAtPort(i);
        auto &mem = child[0]->getMemory();
        auto type = snippet->output(i).get_element_type();
        auto &shape = results[i]->get_shape();
        outputs.push_back(std::make_shared<ngraph::HostTensor>(type, shape, mem.GetPtr()));
    }
    snippet->evaluate(outputs, inputs);
}

REG_MKLDNN_PRIM_FOR(MKLDNNSnippetNode, Subgraph);

