// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_pooling_node.h"

#include "mkldnn_fake_quantize_node.h"
#include "mkldnn_conv_node.h"
#include "mkldnn_concat_node.h"
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <utils/general_utils.h>
#include <cpu_memory_desc_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPoolingNode::MKLDNNPoolingNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    auto maxPoolOp = ngraph::as_type_ptr<ngraph::op::v1::MaxPool>(op);
    auto avgPoolOp = ngraph::as_type_ptr<ngraph::op::v1::AvgPool>(op);
    if (maxPoolOp) {
        algorithm = PoolingMax;
        exclude_pad = false;

        for (int i = 0; i < maxPoolOp->get_strides().size(); i++) {
            stride.push_back(static_cast<ptrdiff_t>(maxPoolOp->get_strides()[i]));
        }
        for (int i = 0; i < maxPoolOp->get_kernel().size(); i++) {
            kernel.push_back(static_cast<ptrdiff_t>(maxPoolOp->get_kernel()[i]));
        }
        for (int i = 0; i < maxPoolOp->get_pads_begin().size(); i++) {
            data_pad_begin.push_back(static_cast<ptrdiff_t>(maxPoolOp->get_pads_begin()[i]));
        }
        for (int i = 0; i < maxPoolOp->get_pads_end().size(); i++) {
            data_pad_end.push_back(static_cast<ptrdiff_t>(maxPoolOp->get_pads_end()[i]));
        }
    } else if (avgPoolOp) {
        algorithm = PoolingAvg;
        exclude_pad = avgPoolOp->get_exclude_pad();

        for (int i = 0; i < avgPoolOp->get_strides().size(); i++) {
            stride.push_back(static_cast<ptrdiff_t>(avgPoolOp->get_strides()[i]));
        }
        for (int i = 0; i < avgPoolOp->get_kernel().size(); i++) {
            kernel.push_back(static_cast<ptrdiff_t>(avgPoolOp->get_kernel()[i]));
        }
        for (int i = 0; i < avgPoolOp->get_pads_begin().size(); i++) {
            data_pad_begin.push_back(static_cast<ptrdiff_t>(avgPoolOp->get_pads_begin()[i]));
        }
        for (int i = 0; i < avgPoolOp->get_pads_end().size(); i++) {
            data_pad_end.push_back(static_cast<ptrdiff_t>(avgPoolOp->get_pads_end()[i]));
        }
    } else {
        IE_THROW(NotImplemented)
                << "CPU Pooling node doesn't support ngraph operation " << op->get_type_name() << " with name " << op->get_friendly_name();
    }
}

std::vector<memory::format_tag> MKLDNNPoolingNode::getAvailableFormatsForDims(const Shape &dims) const {
    if (dims.getRank() == 0)
        return {memory::format_tag::x};
    else if (dims.getRank() == 1)
        return {memory::format_tag::x};
    else if (dims.getRank() == 2)
        return {memory::format_tag::nc};
    else if (dims.getRank() == 3)
        return {memory::format_tag::tnc, memory::format_tag::ntc};
    else if (dims.getRank() == 4)
        return {memory::format_tag::nChw8c, memory::format_tag::nChw16c, memory::format_tag::nhwc, memory::format_tag::nchw};
    else if (dims.getRank() == 5)
        return {memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
    return {memory::format_tag::any};
}

void MKLDNNPoolingNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 1)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();

    inputPrecision = getOriginalInputPrecisionAtPort(0);
    outputPrecision = getOriginalOutputPrecisionAtPort(0);

    // WA: LPT transformation has WA which allows average pooling has I8/U8 output precision instead of FP32,
    // so we explicitly set output precision as FP32
    if (outputPrecision != Precision::I8 && inputPrecision != Precision::BF16) {
        if (getAlgorithm() == PoolingMax) {
            // MKLDNN supports only equal precisions for input and output
            outputPrecision = inputPrecision;
        } else if (getAlgorithm() == PoolingAvg) {
            outputPrecision = Precision::FP32;
        }
    }
    if (inputPrecision == Precision::BF16) {
        outputPrecision = inputPrecision;
    }

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith.back()->getOriginalOutputPrecisionAtPort(0);
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    effective_pad_begin = data_pad_begin;
    effective_pad_end.resize(data_pad_end.size());

    auto parentDims = getParentEdgeAt(0)->getShape().getStaticMklDims();
    auto childDims = getChildEdgeAt(0)->getShape().getStaticMklDims();
    const size_t inputRank = getParentEdgeAt(0)->getShape().getRank();

    if ((inputRank < 4) || (inputRank > 5))
        IE_THROW() << "Pooling layer. Unsupported mode. Only 4D and 5D blobs are supported as input.";

    for (int i = 0; i < effective_pad_end.size(); i++) {
        int krn = kernel[i];
        int src = getParentEdgeAt(0)->getShape().getStaticDims()[2 + i];
        int dst = getChildEdgeAt(0)->getShape().getStaticDims()[2 + i];

        int calc_dst = (src - krn + data_pad_begin[i]) / stride[i] + 1;
        effective_pad_end[i] = (dst - calc_dst) * stride[i];
    }
    if (inputPrecision == Precision::I8 || inputPrecision == Precision::U8) {
        //  We have to extend i8i8_pooling_fwd_t from oneDNN to support BF16 output data type
        if (outputDataType == memory::data_type::bf16)
            outputDataType = memory::data_type::f32;
        // i8 layers supports only ndhwc and nhwc layouts
        const auto in_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(parentDims, inputDataType, inputRank == 5 ?
                                                                 memory::format_tag::ndhwc : memory::format_tag::nhwc);
        const auto out_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(childDims, outputDataType, inputRank == 5 ?
                                                                 memory::format_tag::ndhwc : memory::format_tag::nhwc);
        createDescriptor({ in_candidate.get() }, { out_candidate.get() });
    } else if ((inputRank == 4 || inputRank == 5) && parentDims[1] == 1) {
        // WA. We should force planar layout since it provides better performance
        const auto in_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(parentDims, inputDataType, inputRank == 5 ?
                                                                memory::format_tag::ncdhw : memory::format_tag::nchw);
        const auto out_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(childDims, outputDataType, inputRank == 5 ?
                                                                memory::format_tag::ncdhw : memory::format_tag::nchw);
        createDescriptor({ in_candidate.get() }, { out_candidate.get() });
    } else {
        if (inputDataType != memory::data_type::bf16) {
            inputDataType = memory::data_type::f32;
            outputDataType = memory::data_type::f32;
        }
        // It doesn't support any format
        for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getShape())) {
            const auto in_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(parentDims, inputDataType, format);
            const auto out_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(childDims, outputDataType, format);
            createDescriptor({in_candidate.get()}, {out_candidate.get()});
        }
    }
}

void MKLDNNPoolingNode::createPrimitive() {
    if (prim)
        return;

    mkldnn::primitive_attr attr;
    setPostOps(attr, true);

    auto prim_desc = createPrimitiveDescriptor<pooling_forward::primitive_desc, pooling_forward::desc>(attr);

    prim.reset(new pooling_forward(prim_desc));

    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}};
}

bool MKLDNNPoolingNode::created() const {
    return getType() == Pooling;
}

void MKLDNNPoolingNode::createDescriptor(const std::vector<const MemoryDesc*> &inputDesc,
                                         const std::vector<const MemoryDesc*> &outputDesc) {
    MKLDNNMemoryDesc in_candidate =  MemoryDescUtils::convertToMKLDNNMemoryDesc(*inputDesc[0]);
    MKLDNNMemoryDesc out_candidate = MemoryDescUtils::convertToMKLDNNMemoryDesc(*outputDesc[0]);

    mkldnn::algorithm alg;
    if (algorithm == PoolingAvg) {
        bool not_zero_l = false;
        for (auto lr : data_pad_begin) {
            if (lr) {
                not_zero_l = true;
                break;
            }
        }
        bool not_zero_r = false;
        for (auto pr : data_pad_end) {
            if (pr) {
                not_zero_r = true;
                break;
            }
        }
        if (!exclude_pad && (not_zero_l || not_zero_r))
            alg = mkldnn::algorithm::pooling_avg_include_padding;
        else
            alg = mkldnn::algorithm::pooling_avg_exclude_padding;
    } else if (algorithm == PoolingMax) {
        alg = mkldnn::algorithm::pooling_max;
    } else {
        IE_THROW() << "Unsupported pooling type";
    }

    auto convert = [] (std::vector<ptrdiff_t> orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };
    std::shared_ptr<pooling_forward::desc> desc_ptr(
            new pooling_forward::desc(prop_kind::forward_scoring, alg,
                                      in_candidate, out_candidate,
                                      convert(stride),
                                      convert(kernel),
                                      convert(effective_pad_begin),
                                      convert(effective_pad_end)));

    if (alg == mkldnn::algorithm::pooling_avg_include_padding) {
        // In case of AVG including paddings the norm coeff should be calculated
        // with tacking into account original pads. So we need to restore
        // original values for end paddings.
        //
        // WA. Because mkldnn uses different formula to calculate AVG norm coeff
        //     in compare with Caffe. In mkldnn coeff is always 1/(KH*KW)
        for (int i = 0; i < data_pad_end.size(); i++) {
            if (data_pad_end[i] != effective_pad_end[i])
            desc_ptr->data.padding[1][i] = static_cast<ptrdiff_t>(data_pad_end[i]);
        }
    }

    descs.emplace_back(desc_ptr);
}

void MKLDNNPoolingNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    mkldnn::primitive_attr attr;
    setPostOps(attr);

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
        while (static_cast<bool>(itpd)) {
            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MemoryDescUtils::applyUndefinedOffset(*getSrcMemDesc(itpd, i));
                dataConfig.desc = getSrcMemDesc(itpd, i);
                config.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = MemoryDescUtils::applyUndefinedOffset(*getDstMemDesc(itpd, i));
                dataConfig.desc = getDstMemDesc(itpd, i);
                config.outConfs.push_back(dataConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

void MKLDNNPoolingNode::initDescriptor(const NodeConfig& config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }
    std::vector<const MemoryDesc*> inDescs;
    for (const auto& inConf : config.inConfs)
        inDescs.push_back(inConf.desc.get());
    std::vector<const MemoryDesc*> outDescs;
    for (const auto& outConf : config.outConfs)
        outDescs.push_back(outConf.desc.get());
    createDescriptor({inDescs}, {outDescs});

    mkldnn::primitive_attr attr;
    setPostOps(attr);

    NodeConfig rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;
    for (size_t j = 0; j < descs.size(); j++) {
        const auto &desc = descs[j];
        primitive_desc_iterator itpd;

        itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);

        while (itpd) {
            NodeConfig cfg;
            cfg.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(itpd, i);
                cfg.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(itpd, i);
                cfg.outConfs.push_back(dataConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
            if (selected_count == selectedPrimitiveDescriptorIndex) {
                if (impl_type != selectedPD->getImplementationType()) {
                    IE_THROW() << "Cannot get the original layer configuration!";
                }
                rightConfig = cfg;
            }
            if (j == descs.size() - 1) {
                if (impl_type == selectedPD->getImplementationType()) {
                    rightConfig = config;
                }
            }
            selected_count++;
            if (!itpd.next_impl())
                break;
        }
    }

    if (descs.empty()) {
        const auto& selectedConfig = selectedPD->getConfig();
        if (selectedConfig.inConfs.size() != config.inConfs.size() || selectedConfig.outConfs.size() != config.outConfs.size())
            return;

        for (size_t i = 0; i < selectedConfig.inConfs.size(); i++) {
            if (!selectedConfig.inConfs[i].desc->isCompatible(*config.inConfs[i].desc))
                IE_THROW() << "Incorrect descriptor for node: " << getName();
        }

        for (size_t i = 0; i < selectedConfig.outConfs.size(); i++) {
            if (!selectedConfig.outConfs[i].desc->isCompatible(*config.outConfs[i].desc))
                IE_THROW() << "Incorrect descriptor for node: " << getName();
        }
        rightConfig = config;
    }

    selectedPD->setConfig(rightConfig);
}

void MKLDNNPoolingNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights) {
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

REG_MKLDNN_PRIM_FOR(MKLDNNPoolingNode, Pooling);
