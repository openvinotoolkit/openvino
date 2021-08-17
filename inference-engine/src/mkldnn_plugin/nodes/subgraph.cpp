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

    Precision prec = Precision::FP32;

    NodeConfig config;
    outputShapes[0].getStaticDims();
    config.dynBatchSupport = outputShapes[0].getRank() > 1 && inputShapes[0] == outputShapes[0];
    config.inConfs.resize(inputShapes.size());
    for (auto k = 0; k < inputShapes.size(); k++) {
        config.inConfs[k].inPlace = (!k && canBeInPlace() && inputPrecisions[k] == Precision(Precision::FP32)) ? 0 : -1;
        config.inConfs[k].constant = false;
    }
    PortConfig outDataConfig;
    outDataConfig.inPlace = -1;
    outDataConfig.constant = false;
    config.outConfs.resize(outputShapes.size(), outDataConfig);

    bool dimRanksAreEqual = true;
    for (size_t i = 0; dimRanksAreEqual && i < getParentEdges().size(); i++) {
        for (size_t j = 0; dimRanksAreEqual && j < getChildEdges().size(); j++) {
            if (getParentEdgeAt(i)->getShape().getRank() != getChildEdgeAt(j)->getShape().getRank())
                dimRanksAreEqual = false;
        }
    }

    const size_t ndims = outputShapes[0].getRank();
    const bool isChannelsFirstApplicable = dnnl::impl::utils::one_of(ndims, 1, 2, 4, 5) && dimRanksAreEqual;
    const bool isBlockedApplicable = dnnl::impl::utils::one_of(ndims,  4, 5) && dimRanksAreEqual && !hasBroadcastByC();

    std::vector<LayoutType> tdCreatorTypes;
    if (isChannelsFirstApplicable) {
        tdCreatorTypes.push_back(LayoutType::nspc);
    }
    if (isBlockedApplicable) {
        if (host_isa == dnnl::impl::cpu::x64::avx512_common)
            tdCreatorTypes.push_back(LayoutType::nCsp16c);
        else if (host_isa == dnnl::impl::cpu::x64::avx2)
            tdCreatorTypes.push_back(LayoutType::nCsp8c);
    }
    tdCreatorTypes.push_back(LayoutType::ncsp);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (auto type : tdCreatorTypes) {
        for (auto k = 0; k < inputShapes.size(); k++)
            config.inConfs[k].desc = creatorsMap.at(type)->createUniqueDesc(prec, inputShapes[k].getStaticDims());

        for (auto k = 0; k < outputShapes.size(); k++)
            config.outConfs[k].desc = creatorsMap.at(type)->createUniqueDesc(prec, outputShapes[k].getStaticDims());

        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    }
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
        //return getChildEdgeAt(i)->getMemory().GetDescWithType<BlockedMemoryDesc>().getBlockDims();
        return childEdges[i].lock()->getMemory().GetDescWithType<BlockedMemoryDesc>().getBlockDims();
    };
    auto getOutRank = [getOutBlockedDims](int i) {
        return getOutBlockedDims(i).size();
    };
    size_t max_rank_idx = 0;
    size_t max_rank_val = getOutRank(0);
    for (int i=1; i < childEdges.size(); i++) {
        const auto i_rank_val = getOutRank(i);
        if (max_rank_val < i_rank_val) {
            max_rank_idx = i;
            max_rank_val =  i_rank_val;
        } else if (max_rank_val == i_rank_val) {
            const auto max_rank_dims = getOutBlockedDims(max_rank_idx);
            const auto i_dims = getOutBlockedDims(i);
            for (int j = 0; j < max_rank_val; j++) {
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
    tensorRank = std::max(static_cast<size_t>(rank6D), outBlockingDesc_maxRank.getBlockDims().size());

    auto initDims = [this, config, &outBlockingDesc_maxRank](size_t tensorRank) {
        // assume all input sizes are even
        size_t inputNum = getParentEdges().size();

        dims_in.resize(inputNum);
        for (int i = 0; i < inputNum; i++) {
            dims_in[i].resize(tensorRank, 1);
        }

        const auto outOrder = outBlockingDesc_maxRank.getOrder();
        for (int i = 0; i < inputNum; i++) {
            auto inBlockingDesc = getParentEdgeAt(i)->getMemory().GetDescWithType<BlockedMemoryDesc>();
            size_t rank = inBlockingDesc.getBlockDims().size();

            // WA to normalize blocked and planar layouts
            // not actual thought, since [§] doesn't support mixed layouts yet
            auto inOrder = inBlockingDesc.getOrder();
            size_t startOff = outOrder.size() != outBlockingDesc_maxRank.getShape().getRank() &&
                              outOrder.back() != inOrder.back() ? 1 : 0;
            for (int j = 0; j < rank; j++) {
                dims_in[i][dims_in[i].size() - 1 - j - startOff]
                = inBlockingDesc.getBlockDims()[rank - 1 - j];
            }
        }

        // assume all output sizes are even
        size_t outputNum = config.outConfs.size();//getChildEdges().size();

        dims_out.resize(outputNum);
        for (int i = 0; i < outputNum; i++) {
            dims_out[i].resize(tensorRank, 1);
        }

        for (int i = 0; i < outputNum; i++) {
            auto outBlockingDesc = getChildEdgeAt(i)->getMemory().GetDescWithType<BlockedMemoryDesc>();
            size_t rank = outBlockingDesc.getBlockDims().size();

            for (int j = 0; j < rank; j++) {
                dims_out[i][dims_out[i].size() - 1 - j]
                = outBlockingDesc.getBlockDims()[rank - 1 - j];
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

    auto find_dims_to_collapse = [this, config, &outBlockingDesc_maxRank]() -> int {
        int collapsedDims = 0;
        size_t minimalConcurrency = parallel_get_max_threads();
        size_t minimalJitWorkAmount = 256;
        size_t currentJitWorkAmount = dims_out[max_rank_out_desc_idx].back();
        while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount &&
               // we shouldn't collapse batch dimension in case dynamic batch is enabled
               (!isDynBatchEnabled || (outBlockingDesc_maxRank.getBlockDims().size() - collapsedDims > 2))) {
            if (dims_out[max_rank_out_desc_idx].size() - collapsedDims - 2 < 0)
                break;

            bool canCollapse = true;
            for (int i = 0; i < dims_in.size(); i++) {
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
        if (tileRank > 1) {
            sch_dims[1] = dims_out[max_rank_out_desc_idx][tensorRank - 2];
            dims_out[max_rank_out_desc_idx][tensorRank - 2] = 1;

            // update offsets for tile 2D because loaders have ptr shifts in some cases and stores have always ptrs shifts
            for (auto i = 0; i < offsets_in.size(); i++) {
                int64_t offset = offsets_in[i][tensorRank - 2];
                if ((offset > dataSize) || (offset == 0 && dims_in[i].back() != 1)) {
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

    initDims(tensorRank);

    fullWorkAmount = 1;
    for (int i = 0; i < dims_out[max_rank_out_desc_idx].size(); i++) {
        fullWorkAmount *= dims_out[max_rank_out_desc_idx][i];
    }

    isDynBatchEnabled = config.dynBatchSupport;

    const int collapsedDims = find_dims_to_collapse();
    batchDimIdx = tensorRank - outBlockingDesc_maxRank.getBlockDims().size() + collapsedDims;
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
        ngraph::Shape shape(blockedDesc.getBlockDims());
        ngraph::AxisVector blocking(blockedDesc.getOrder());
        ngraph::element::Type precision = (blockedDesc.getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
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
        ngraph::Shape shape(blockedDesc.getBlockDims());
        ngraph::AxisVector blocking(blockedDesc.getOrder());
        ngraph::element::Type precision = (blockedDesc.getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
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
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto & parents = getParentEdgesAtPort(i);
        auto &mem = parents[0]->getMemory();
        auto type = snippet->input(i).get_element_type();
        auto &shape = params[i]->get_shape();
        inputs.push_back(std::make_shared<ngraph::HostTensor>(type, shape, mem.GetPtr()));
    }

    ngraph::HostTensorVector outputs;
    auto results = snippet->get_body()->get_results();
    for (size_t i = 0; i < outputShapes.size(); i++) {
        auto & child = getChildEdgesAtPort(i);
        auto &mem = child[0]->getMemory();
        auto type = snippet->output(i).get_element_type();
        auto &shape = results[i]->get_shape();
        outputs.push_back(std::make_shared<ngraph::HostTensor>(type, shape, mem.GetPtr()));
    }
    snippet->evaluate(outputs, inputs);
}

REG_MKLDNN_PRIM_FOR(MKLDNNSnippetNode, Subgraph);

