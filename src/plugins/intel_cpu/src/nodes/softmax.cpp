// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax.h"

#include <memory_desc/cpu_memory_desc_utils.h>

#include <shape_inference/shape_inference_pass_through.hpp>
#include <string>

#include "common/primitive_hashing_utils.hpp"
#include "dnnl_extension_utils.h"
#include "dnnl_types.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "openvino/opsets/opset1.hpp"

using namespace dnnl;

namespace ov::intel_cpu::node {
namespace {

struct SoftmaxKey {
    DnnlMemoryDescCPtr inp0;
    impl_desc_type implType;
    size_t axis;
    dnnl::primitive_attr attr;

    [[nodiscard]] size_t hash() const;
    bool operator==(const SoftmaxKey& rhs) const;
};

size_t SoftmaxKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    seed = hash_combine(seed, get_md_hash(*inp0->getDnnlDesc().get()));
    seed = hash_combine(seed, implType);
    seed = hash_combine(seed, axis);
    return seed;
}

bool SoftmaxKey::operator==(const SoftmaxKey& rhs) const {
    bool retVal = true;
    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }

    retVal = retVal && implType == rhs.implType && axis == rhs.axis;
    return retVal;
}

}  // namespace

bool SoftMax::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::as_type_ptr<const ov::opset1::Softmax>(op)) {
            errorMessage = "Only opset1 Softmax operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

SoftMax::SoftMax(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    axis = ov::as_type_ptr<ov::op::v1::Softmax>(op)->get_axis();
}

void SoftMax::getSupportedDescriptors() {
    if (descs.size()) {
        return;
    }

    ov::element::Type precision = getOriginalInputPrecisionAtPort(0);
    if (!one_of(precision, ov::element::f32, ov::element::bf16, ov::element::f16)) {
        precision = ov::element::f32;
    }
    auto inputDataType = DnnlExtensionUtils::ElementTypeToDataType(precision);

    if (getParentEdges().size() != 1) {
        THROW_CPU_NODE_ERR("Incorrect number of input edges");
    }
    if (!getChildEdges().size()) {
        THROW_CPU_NODE_ERR("Incorrect number of output edges");
    }

    const auto& inShape = getInputShapeAtPort(0);
    if (inShape.getRank() == 3) {
        auto in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inShape, inputDataType, memory::format_tag::abc);
        createDescriptor({in_candidate}, {});
    }

    for (auto format : getAvailableFormatsForDims(inShape)) {
        auto in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inShape, inputDataType, format);

        if (in_candidate->blocksExtended()) {
            continue;
        }

        createDescriptor({in_candidate}, {});
    }
}

bool SoftMax::created() const {
    return getType() == Type::Softmax;
}

Node::AttrPtr SoftMax::initPrimitiveAttr() {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());
    (*attr).set_scratchpad_mode(dnnl::scratchpad_mode::user);

    return attr;
}

void SoftMax::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr) {
        THROW_CPU_NODE_ERR("Preferable primitive descriptor is not set.");
    }
    auto config = selected_pd->getConfig();
    if (isDynamicNode()) {
        auto outMemDesc = config.outConfs[0].getMemDesc();
        config.outConfs[0].setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(outMemDesc),
                                      BlockedMemoryDesc::FULL_MASK);
    } else {
        if (config.inConfs.size() != 1 || config.outConfs.size() != 1 ||
            (config.inConfs[0].getMemDesc()->isDefined() && config.outConfs[0].getMemDesc()->isDefined() &&
             !config.outConfs[0].getPortDesc()->isCompatible(*config.inConfs[0].getPortDesc()))) {
            THROW_CPU_NODE_ERR("has incorrect selected config!");
        }

        config.inConfs[0].setMemDesc(getConsistentInputDesc(config, 0)->getMemDesc());
        config.outConfs[0].setMemDesc(config.inConfs[0].getMemDesc());
    }
    initDescriptor(config);
}

void SoftMax::createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                               const std::vector<MemoryDescPtr>& outputDesc) {
    auto inpDesc = inputDesc[0]->isDefined() ? inputDesc[0] : MemoryDescUtils::makeDummyDesc(*inputDesc[0]);
    DnnlMemoryDescPtr definedInpMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(inpDesc);
    auto in_candidate = definedInpMemDesc->getDnnlDesc();

    auto attr = initPrimitiveAttr();

    auto desc = softmax_forward::primitive_desc(getEngine(),
                                                prop_kind::forward_inference,
                                                algorithm::softmax_accurate,
                                                in_candidate,
                                                in_candidate,
                                                axis,
                                                *attr,
                                                true);

    if (desc) {
        descs.emplace_back(desc);
    }
}

void SoftMax::prepareParams() {
    auto inpDesc = getParentEdgeAt(0)->getMemory().getDescWithType<DnnlMemoryDesc>();
    const NodeDesc* selected_pd = getSelectedPrimitiveDescriptor();

    if (selected_pd == nullptr) {
        THROW_CPU_NODE_ERR("Preferable primitive descriptor is not set.");
    }

    auto attr = initPrimitiveAttr();

    SoftmaxKey key = {inpDesc, selected_pd->getImplementationType(), axis, *attr};
    auto engine = getEngine();

    auto builder = [&engine](const SoftmaxKey& key) -> executorPtr {
        auto prim_desc = softmax_forward::primitive_desc(engine,
                                                         prop_kind::forward_inference,
                                                         algorithm::softmax_accurate,
                                                         key.inp0->getDnnlDesc(),
                                                         key.inp0->getDnnlDesc(),
                                                         key.axis,
                                                         key.attr,
                                                         true);

        primitive_desc_iterator itpd = prim_desc;

        auto itpd_first = itpd;
        while (itpd) {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
            if (impl_type == key.implType ||
                // At least for oneDNN v2.4 the softmax primitive is optimized for the cases where the dimension of the
                // softmax axis is physically dense. There could be situations where it is not possible to detect the
                // optimized case in advance in case of dynamic shapes, but in runtime the shape could be suitable for
                // the optimized implementation, so we have to select the optimized one.
                (ref_any == key.implType && (impl_type & jit))) {
                prim_desc = itpd.get();
                break;
            }
            if (!itpd.next_impl()) {
                prim_desc = itpd_first.get();
                break;
            }
        }
        return std::make_shared<DnnlExecutor>(prim_desc);
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    execPtr = result.first;
    if (!execPtr) {
        THROW_CPU_NODE_ERR("Primitive descriptor was not found.");
    }

    auto scratchpadMem = getScratchPadMem(execPtr->getScratchPadDesc());

    primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMem->getPrimitive();
    primArgs[DNNL_ARG_SRC] = getSrcMemoryAtPort(0)->getPrimitive();
    primArgs[DNNL_ARG_DST] = getDstMemoryAtPort(0)->getPrimitive();
#ifdef CPU_DEBUG_CAPS
    auto pd = execPtr->getPrimitiveDesc();
    DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
#endif
}

void SoftMax::execute(const dnnl::stream& strm) {
    if (execPtr) {
        execPtr->exec(primArgs, strm);
    } else {
        THROW_CPU_NODE_ERR("doesn't have an initialized executor");
    }
}

void SoftMax::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

}  // namespace ov::intel_cpu::node
