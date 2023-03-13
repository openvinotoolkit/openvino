// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling.h"

#include "fake_quantize.h"
#include "conv.h"
#include "concat.h"
#include <string>
#include <vector>
#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>
#include <utils/general_utils.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <common/primitive_hashing_utils.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct PoolingKey {
    DnnlMemoryDescCPtr inp;
    DnnlMemoryDescCPtr out;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> kernel;
    /// Effective padding. Used to define correct output shape by oneDNN
    /// reshape formula: (iw - kernel + pad_l + pad_r) / strides[i - 2] + 1
    /// should be passed into pooling desc constructor.
    std::vector<ptrdiff_t> effective_pad_begin;
    std::vector<ptrdiff_t> effective_pad_end;
    /// Effective dilation. Used to define correct dilation for OneDNN.
    /// For OneDNN default dilation is vector of zero
    std::vector<ptrdiff_t> effective_dilation;
    std::vector<ptrdiff_t> data_pad_end;
    dnnl::primitive_attr attr;
    algorithm alg;
    impl_desc_type implType;

    size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        seed = hash_combine(seed, get_md_hash(inp->getDnnlDesc().data));
        seed = get_vector_hash(seed, stride);
        seed = get_vector_hash(seed, kernel);
        seed = get_vector_hash(seed, effective_pad_begin);
        seed = get_vector_hash(seed, effective_pad_end);
        seed = get_vector_hash(seed, effective_dilation);
        seed = get_vector_hash(seed, data_pad_end);
        seed = hash_combine(seed, get_md_hash(out->getDnnlDesc().data));
        seed = hash_combine(seed, get_attr_hash(*attr.get()));
        seed = hash_combine(seed, alg);
        seed = hash_combine(seed, implType);
        return seed;
    }

    bool operator==(const PoolingKey& rhs) const {
        bool result = true;
        if (inp != rhs.inp) {
            result = result && inp && rhs.inp && (inp->getDnnlDesc() == rhs.inp->getDnnlDesc());
        }

        if (out != rhs.out) {
            result = result && out && rhs.out && (out->getDnnlDesc() == rhs.out->getDnnlDesc());
        }

        result = result && stride == rhs.stride && kernel == rhs.kernel &&
                 effective_pad_begin == rhs.effective_pad_begin && effective_pad_end == rhs.effective_pad_end &&
                 effective_dilation == rhs.effective_dilation && data_pad_end == rhs.data_pad_end &&
                 *attr.get() == *rhs.attr.get() && alg == rhs.alg && implType == rhs.implType;
        return result;
    }
};

std::shared_ptr<pooling_v2_forward::desc> createDescriptorHelper(const dnnl::memory::desc& in_candidate,
                                                                 const dnnl::memory::desc& out_candidate,
                                                                 const dnnl::algorithm alg,
                                                                 const std::vector<ptrdiff_t>& stride,
                                                                 const std::vector<ptrdiff_t>& kernel,
                                                                 const std::vector<ptrdiff_t>& effective_pad_begin,
                                                                 const std::vector<ptrdiff_t>& effective_pad_end,
                                                                 const std::vector<ptrdiff_t>& effective_dilation,
                                                                 const std::vector<ptrdiff_t>& data_pad_end) {
    if (alg == dnnl::algorithm::undef) {
        IE_THROW() << "Unsupported pooling type";
    }

    auto convert = [](std::vector<ptrdiff_t> orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };
    std::shared_ptr<pooling_v2_forward::desc> desc_ptr(new pooling_v2_forward::desc(prop_kind::forward_scoring,
                                                                                    alg,
                                                                                    in_candidate,
                                                                                    out_candidate,
                                                                                    convert(stride),
                                                                                    convert(kernel),
                                                                                    convert(effective_dilation),
                                                                                    convert(effective_pad_begin),
                                                                                    convert(effective_pad_end)));

    if (alg == dnnl::algorithm::pooling_avg_include_padding) {
        // In case of AVG including paddings the norm coeff should be calculated
        // with tacking into account original pads. So we need to restore
        // original values for end paddings.
        //
        // WA. Because onednn uses different formula to calculate AVG norm coeff
        //     in compare with Caffe. In onednn coeff is always 1/(KH*KW)
        for (int i = 0; i < data_pad_end.size(); i++) {
            if (data_pad_end[i] != effective_pad_end[i])
                desc_ptr->data.padding[1][i] = static_cast<ptrdiff_t>(data_pad_end[i]);
        }
    }

    return desc_ptr;
}

}  // namespace

bool Pooling::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (ov::is_type<const ov::op::v8::MaxPool>(op)) {
            if (!op->get_output_target_inputs(1).empty()) {
                errorMessage = "MaxPool from opset8 is supported only with one output";
                return false;
            }
        } else if (!ov::is_type<const ov::op::v1::MaxPool>(op) && !ov::is_type<const ov::op::v1::AvgPool>(op)) {
            errorMessage = "MaxPool and AvgPool from opset1 and MaxPool from opset8 are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Pooling::Pooling(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto get_attributes = [](std::vector<ptrdiff_t>& internal_attribute, const std::vector<size_t> external_attribute) {
        for (size_t i = 0; i < external_attribute.size(); i++) {
            internal_attribute.push_back(static_cast<ptrdiff_t>(external_attribute[i]));
        }
    };

    if (auto maxPoolOp_v8 = ov::as_type_ptr<const ov::op::v8::MaxPool>(op)) {
        isMaxPool8 = true;
        algorithm = Algorithm::PoolingMax;
        exclude_pad = false;

        get_attributes(dilation, maxPoolOp_v8->get_dilations());
        get_attributes(stride, maxPoolOp_v8->get_strides());
        get_attributes(kernel, maxPoolOp_v8->get_kernel());
        get_attributes(data_pad_begin, maxPoolOp_v8->get_pads_begin());
        get_attributes(data_pad_end, maxPoolOp_v8->get_pads_end());

        auto_pad = (maxPoolOp_v8->get_auto_pad() == ov::op::PadType::SAME_LOWER || maxPoolOp_v8->get_auto_pad() == ov::op::PadType::SAME_UPPER);
    } else if (auto maxPoolOp_v1 = ov::as_type_ptr<const ov::op::v1::MaxPool>(op)) {
        algorithm = Algorithm::PoolingMax;
        exclude_pad = false;

        get_attributes(stride, maxPoolOp_v1->get_strides());
        get_attributes(kernel, maxPoolOp_v1->get_kernel());
        get_attributes(data_pad_begin, maxPoolOp_v1->get_pads_begin());
        get_attributes(data_pad_end, maxPoolOp_v1->get_pads_end());
        dilation.resize(kernel.size(), 1);

        auto_pad = (maxPoolOp_v1->get_auto_pad() == ov::op::PadType::SAME_LOWER || maxPoolOp_v1->get_auto_pad() == ov::op::PadType::SAME_UPPER);
    } else if (auto avgPoolOp = ov::as_type_ptr<const ov::op::v1::AvgPool>(op)) {
        algorithm = Algorithm::PoolingAvg;
        exclude_pad = avgPoolOp->get_exclude_pad();

        get_attributes(stride, avgPoolOp->get_strides());
        get_attributes(kernel, avgPoolOp->get_kernel());
        get_attributes(data_pad_begin, avgPoolOp->get_pads_begin());
        get_attributes(data_pad_end, avgPoolOp->get_pads_end());
        dilation.resize(kernel.size(), 1);

        auto_pad = (avgPoolOp->get_auto_pad() == ov::op::PadType::SAME_LOWER || avgPoolOp->get_auto_pad() == ov::op::PadType::SAME_UPPER);
    }
}

std::vector<memory::format_tag> Pooling::getAvailableFormatsForDims(const Shape &dims) const {
    if (dims.getRank() == 0)
        return {memory::format_tag::x};
    else if (dims.getRank() == 1)
        return {memory::format_tag::x};
    else if (dims.getRank() == 2)
        return {memory::format_tag::nc};
    else if (dims.getRank() == 3)
        return { memory::format_tag::nCw8c, memory::format_tag::nCw16c, memory::format_tag::nwc, memory::format_tag::ncw};
    else if (dims.getRank() == 4)
        return {memory::format_tag::nChw8c, memory::format_tag::nChw16c, memory::format_tag::nhwc, memory::format_tag::nchw};
    else if (dims.getRank() == 5)
        return {memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
    return {memory::format_tag::any};
}

void Pooling::initEffectiveAttributes(const Shape &inShape, const Shape &outShape) {
    effective_pad_begin = data_pad_begin;
    effective_pad_end.resize(data_pad_end.size());
    effective_dilation.resize(dilation.size(), 0);

    const auto &inDims = inShape.getStaticDims();
    const auto &outDims = outShape.getStaticDims();

    for (int i = 0; i < effective_pad_end.size(); i++) {
        int krn = kernel[i];
        int dil = dilation[i];
        int src = inDims[2 + i];
        int dst = outDims[2 + i];

        int calc_dst = (src - (1 + (krn  - 1) * dil) + data_pad_begin[i]) / stride[i] + 1;
        effective_pad_end[i] = (dst - calc_dst) * stride[i];
        effective_dilation[i] = dil - 1;
    }
}

void Pooling::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 1)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();

    InferenceEngine::Precision inputPrecision = getOriginalInputPrecisionAtPort(0);
    InferenceEngine::Precision outputPrecision = getOriginalOutputPrecisionAtPort(0);

    // WA: LPT transformation has WA which allows average pooling has I8/U8 output precision instead of FP32,
    // so we explicitly set output precision as FP32
    if (outputPrecision != Precision::I8 && inputPrecision != Precision::BF16) {
        if (getAlgorithm() == Algorithm::PoolingMax) {
            // oneDNN supports only equal precisions for input and output
            outputPrecision = inputPrecision;
        } else if (getAlgorithm() == Algorithm::PoolingAvg) {
            outputPrecision = Precision::FP32;
        }
    }
    if (inputPrecision == Precision::BF16) {
        outputPrecision = inputPrecision;
    }

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith.back()->getOriginalOutputPrecisionAtPort(0);
    }

    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(outputPrecision);

    const auto &parentShape = getInputShapeAtPort(0);
    const auto &childShape = getOutputShapeAtPort(0);
    const size_t inputRank = getInputShapeAtPort(0).getRank();

    if ((inputRank < 3) || (inputRank > 5))
        IE_THROW() << "Pooling layer. Unsupported mode. Only 3D, 4D and 5D blobs are supported as input.";

    inShape = MemoryDescUtils::makeDummyShape(parentShape);
    if (isDynamicNode()) {
        const auto& origDims = parentShape.getDims();
        const auto& origMaxDims = parentShape.getMaxDims();

        auto inDims = inShape.getStaticDims();
        for (size_t i = 0; i < inDims.size() - 2; i++) {
            if (origDims[i + 2] == Shape::UNDEFINED_DIM) {
                inDims[i + 2] = std::min<Dim>(origMaxDims[i + 2], std::max<Dim>(inDims[i + 2], kernel[i]));
            }
        }
        inShape = Shape(inDims);
    }

    initEffectiveAttributes(inShape,
                            MemoryDescUtils::makeDummyShape(childShape));

    if (inputPrecision == Precision::I8 || inputPrecision == Precision::U8) {
        //  We have to extend i8i8_pooling_fwd_t from oneDNN to support BF16 output data type
        if (outputDataType == memory::data_type::bf16)
            outputDataType = memory::data_type::f32;
        // i8 layers supports only ndhwc and nhwc layouts
        const auto in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(parentShape, inputDataType, inputRank == 3 ?
                                  memory::format_tag::nwc : (inputRank == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc));
        const auto out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(childShape, outputDataType, inputRank == 3 ?
                                   memory::format_tag::nwc : (inputRank == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc));
        createDescriptor({ in_candidate }, { out_candidate });
    } else if ((inputRank == 3 || inputRank == 4 || inputRank == 5) && parentShape.getDims()[1] == 1) {
        // WA. We should force planar layout since it provides better performance
        const auto in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(parentShape, inputDataType, inputRank == 3 ?
                                  memory::format_tag::ncw : (inputRank == 4 ? memory::format_tag::nchw : memory::format_tag::ncdhw));
        const auto out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(childShape, outputDataType, inputRank == 3 ?
                                   memory::format_tag::ncw : (inputRank == 4 ? memory::format_tag::nchw : memory::format_tag::ncdhw));
        createDescriptor({ in_candidate }, { out_candidate });
    } else {
        if (inputDataType != memory::data_type::bf16) {
            inputDataType = memory::data_type::f32;
            outputDataType = memory::data_type::f32;
        }
        // It doesn't support any format
        for (auto format : getAvailableFormatsForDims(getInputShapeAtPort(0))) {
            const auto in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(parentShape, inputDataType, format);
            const auto out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(childShape, outputDataType, format);
            createDescriptor({in_candidate}, {out_candidate});
        }
    }
}

void Pooling::prepareParams() {
    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW()  << "Pooling node with name '" << getName() << "' did not set preferable primitive descriptor";

    AttrPtr attr;
    if (isDynamicNode()) {
        if (!pAttr) {
            pAttr = initPrimitiveAttr();
        }
        attr = pAttr;
    } else {
        attr = initPrimitiveAttr();
    }

    auto inDesc = getParentEdgesAtPort(0)[0]->getMemory().GetDescWithType<DnnlMemoryDesc>();
    auto outDesc = getChildEdgesAtPort(0)[0]->getMemory().GetDescWithType<DnnlMemoryDesc>();

    if (isDynamicNode()) {
        if (auto_pad) {
            data_pad_begin = shapeInference->get_pads_begin();
            data_pad_end = shapeInference->get_pads_end();
        }
        initEffectiveAttributes(inDesc->getShape(), outDesc->getShape());
    }

    dnnl::algorithm alg = getPoolingAlgorithm();
    PoolingKey key = {inDesc,
                      outDesc,
                      stride,
                      kernel,
                      effective_pad_begin,
                      effective_pad_end,
                      effective_dilation,
                      data_pad_end,
                      *attr,
                      alg,
                      selected_pd->getImplementationType()};
    auto engine = getEngine();
    auto builder = [&engine](const PoolingKey& key) -> dnnl::primitive {
        auto desc_ptr = createDescriptorHelper(key.inp->getDnnlDesc(),
                                               key.out->getDnnlDesc(),
                                               key.alg,
                                               key.stride,
                                               key.kernel,
                                               key.effective_pad_begin,
                                               key.effective_pad_end,
                                               key.effective_dilation,
                                               key.data_pad_end);
        DnnlDesriptor desc{desc_ptr};
        primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine, key.attr);
        pooling_v2_forward::primitive_desc prim_desc = itpd.get();
        while (static_cast<bool>(itpd)) {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            if (impl_type == key.implType) {
                prim_desc = itpd.get();
                break;
            }
            if (!itpd.next_impl())
                break;
        }
        return pooling_v2_forward(prim_desc);
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    prim = result.first;

    auto pd = prim.get_primitive_desc();
    auto scratchpadMem = getScratchPadMem(pd);
    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}, {DNNL_ARG_SCRATCHPAD, scratchpadMem->GetPrimitive()}};

    Node::appendPostOpArgs(*attr, primArgs, postOpsArgs);
}

void Pooling::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Pooling::created() const {
    return getType() == Type::Pooling;
}

dnnl::algorithm Pooling::getPoolingAlgorithm() const {
    if (algorithm == Algorithm::PoolingAvg) {
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
            return dnnl::algorithm::pooling_avg_include_padding;
        else
            return dnnl::algorithm::pooling_avg_exclude_padding;
    } else if (algorithm == Algorithm::PoolingMax) {
        return dnnl::algorithm::pooling_max;
    } else {
        return dnnl::algorithm::undef;
    }
}

std::shared_ptr<pooling_v2_forward::desc> Pooling::createDescriptorInternal(
    const dnnl::memory::desc& in_candidate,
    const dnnl::memory::desc& out_candidate,
    const dnnl::algorithm alg) const {
    return createDescriptorHelper(in_candidate,
                                  out_candidate,
                                  alg,
                                  stride,
                                  kernel,
                                  effective_pad_begin,
                                  effective_pad_end,
                                  effective_dilation,
                                  data_pad_end);
}

void Pooling::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                                         const std::vector<MemoryDescPtr> &outputDesc) {
    auto inDesc = inputDesc[0]->isDefined() ? inputDesc[0] : inputDesc[0]->cloneWithNewDims(inShape.getStaticDims());
    auto dnnlInDesc = MemoryDescUtils::convertToDnnlMemoryDesc(inDesc);
    auto in_candidate = dnnlInDesc->getDnnlDesc();

    auto outDesc = outputDesc[0];
    if (!outDesc->isDefined()) {
        auto outDims = shapeInferGeneric({Shape(inDesc->getShape().getStaticDims())});
        outDesc = outDesc->cloneWithNewDims(outDims[0]);
        if (auto_pad) {
            data_pad_begin = shapeInference->get_pads_begin();
            data_pad_end = shapeInference->get_pads_end();
        }
        initEffectiveAttributes(inDesc->getShape(), outDesc->getShape());
    }
    auto dnnlOutDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*outDesc);
    auto out_candidate = dnnlOutDesc.getDnnlDesc();

    auto desc_ptr = createDescriptorInternal(in_candidate, out_candidate, getPoolingAlgorithm());
    descs.emplace_back(desc_ptr);
}

void Pooling::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    dnnl::primitive_attr attr;
    setPostOps(attr);

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
        while (static_cast<bool>(itpd)) {
            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig dataConfig;
                dataConfig.inPlace(-1);
                dataConfig.constant(false);
                dataConfig.setMemDesc(getSrcMemDesc(itpd, i));

                config.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig dataConfig;
                dataConfig.inPlace(canBeInPlace() ? 0 : -1);
                dataConfig.constant(false);
                dataConfig.setMemDesc(getDstMemDesc(itpd, i));

                config.outConfs.push_back(dataConfig);
            }

            // CPU plugin doesn't support second output of MaxPool-8, but anyway we should have out config for second port as stub
            if (isMaxPool8) {
                auto& creatorsMap = BlockedDescCreator::getCommonCreators();
                PortConfig dataConfig;
                dataConfig.inPlace(-1);
                dataConfig.constant(false);
                dataConfig.setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(config.outConfs.front().getMemDesc()->getPrecision(),
                                                                                         getOutputShapeAtPort(1)));

                config.outConfs.push_back(dataConfig);
            }

            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

void Pooling::initDescriptor(const NodeConfig& config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }
    std::vector<MemoryDescPtr> inDescs;
    for (const auto& inConf : config.inConfs)
        inDescs.push_back(inConf.getMemDesc());
    std::vector<MemoryDescPtr> outDescs;
    for (const auto& outConf : config.outConfs)
        outDescs.push_back(outConf.getMemDesc());
    createDescriptor(inDescs, outDescs);

    dnnl::primitive_attr attr;
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
                dataConfig.inPlace(canBeInPlace() ? 0 : -1);
                dataConfig.constant(false);
                dataConfig.setMemDesc(getSrcMemDesc(itpd, i));
                cfg.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig dataConfig;
                dataConfig.inPlace(-1);
                dataConfig.constant(false);
                dataConfig.setMemDesc(getDstMemDesc(itpd, i));
                cfg.outConfs.push_back(dataConfig);
            }

            // CPU plugin doesn't support second output of MaxPool-8, but anyway we should have out config for second port as stub
            if (isMaxPool8) {
                auto& creatorsMap = BlockedDescCreator::getCommonCreators();
                PortConfig dataConfig;
                dataConfig.inPlace(-1);
                dataConfig.constant(false);
                dataConfig.setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(cfg.outConfs.front().getMemDesc()->getPrecision(),
                                                                                         getOutputShapeAtPort(1)));

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
            if (!selectedConfig.inConfs[i].getPortDesc()->isCompatible(*config.inConfs[i].getPortDesc()))
                IE_THROW() << "Incorrect descriptor for node: " << getName();
        }

        for (size_t i = 0; i < selectedConfig.outConfs.size(); i++) {
            if (!selectedConfig.outConfs[i].getPortDesc()->isCompatible(*config.outConfs[i].getPortDesc()))
                IE_THROW() << "Incorrect descriptor for node: " << getName();
        }
        rightConfig = config;
    }

    selectedPD->setConfig(rightConfig);
}

Node::AttrPtr Pooling::initPrimitiveAttr() {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

    setPostOps(*attr);

    (*attr).set_scratchpad_mode(dnnl::scratchpad_mode::user);

    return attr;
}

void Pooling::setPostOps(dnnl::primitive_attr &attr) {
    dnnl::post_ops ops;

    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsArgs);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

}  // namespace node
}   // namespace intel_cpu
}   // namespace ov
