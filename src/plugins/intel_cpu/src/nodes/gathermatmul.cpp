// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gathermatmul.h"

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_map>

#include "common/blocked_desc_creator.h"
#include "config.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "node.h"
#include "node_config.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"
#include "shape_inference/custom/gathermatmul.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_cpu::node {

bool GatherMatmul::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const bool isGatherMatmul = ov::is_type<ov::op::internal::GatherMatmul>(op);
        const bool isGatherMatmulCompressed = ov::is_type<ov::op::internal::GatherMatmulCompressed>(op);

        if (!isGatherMatmul && !isGatherMatmulCompressed) {
            errorMessage = "Only GatherMatmul and GatherMatmulCompressed operations are supported. Got: " +
                           std::string(op->get_type_info().name);
            return false;
        }

        if (!ov::op::util::is_on_path<ov::op::v0::Constant>(op->input_value(WEIGHTS))) {
            errorMessage = "Only constant weights are supported for GatherMatmul operation";
            return false;
        }

        if (isGatherMatmulCompressed) {
            if (op->get_input_size() > WEIGHT_SCALES) {
                if (!ov::op::util::is_on_path<ov::op::v0::Constant>(op->input_value(WEIGHT_SCALES))) {
                    errorMessage = "Only constant weight scales are supported for GatherMatmul operation";
                    return false;
                }
            }
            if (op->get_input_size() > WEIGHT_ZERO_POINTS) {
                if (!ov::op::util::is_on_path<ov::op::v0::Constant>(op->input_value(WEIGHT_ZERO_POINTS))) {
                    errorMessage = "Only constant weight zero points are supported for GatherMatmul operation";
                    return false;
                }
            }
        }

        if (op->get_input_size() > BIAS) {
            const auto& biasInput = op->input_value(BIAS);
            if (biasInput.get_element_type() != ov::element::dynamic) {
                if (!ov::op::util::is_on_path<ov::op::v0::Constant>(biasInput)) {
                    errorMessage = "Only constant bias is supported for GatherMatmul operation";
                    return false;
                }
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool GatherMatmul::isSupportedCompressedOperation([[maybe_unused]] const std::shared_ptr<ov::Node>& op,
                                                  [[maybe_unused]] size_t IC,
                                                  [[maybe_unused]] size_t OC,
                                                  [[maybe_unused]] size_t G,
                                                  [[maybe_unused]] const Config& config) noexcept {
#ifdef OPENVINO_ARCH_X86_64
    try {
        std::string errorMessage;
        if (!isSupportedOperation(op, errorMessage)) {
            return false;
        }

        if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
            return false;
        }

        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) &&
            config.inferencePrecision == ov::element::bf16) {
            // OneDNN AMX IP implementation has limited shapes support due to performance considerations. As a
            // current solution conditions below are copied from OneDNN to make sure correct IP impl will be
            // used since fallback one doesn't support weights decompression feature.
            constexpr size_t simdWidth = 16;
            constexpr size_t vnniFactor = 2;
            constexpr size_t maxSize = 512;
            constexpr size_t amxRow = vnniFactor * simdWidth;

            if ((IC <= amxRow && OC <= amxRow) || (IC <= maxSize && OC <= maxSize && IC % amxRow != 0)) {
                return false;
            }
        }

        if (IC % G != 0) {
            return false;  // sanity check IC must be evenly divided by the group size
        }

        if (IC / G < 4) {
            return false;  // minimal group size should be 4
        }

        if (OC == 1) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
#else
    return false;
#endif
}

ov::element::TypeVector GatherMatmul::getSupportedCompressedWeightsTypes([[maybe_unused]] bool apply_fp8) {
    using ov::element::Type_t;
#ifdef OPENVINO_ARCH_X86_64
    return {Type_t::u8, Type_t::i8, Type_t::u4, Type_t::i4};
#else
    return {};
#endif
}

ov::element::TypeVector GatherMatmul::getSupportedCompressedActivationsTypes() {
    using ov::element::Type_t;
    // @todo enable for bf16 as well
    // after EnforceInferencePrecision is replaced with ConvertPrecision
    return {Type_t::f32};
}

GatherMatmul::GatherMatmul(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, GatherMatmulShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    m_isCompressed = ov::is_type<ov::op::internal::GatherMatmulCompressed>(op);

    m_atoi[ARG_SRC] = DATA;
    m_atoi[ARG_WEI] = WEIGHTS;
    m_atoi[ARG_SRC_1] = INDICES;
    m_atoi[ARG_BIAS] = BIAS;
    if (m_isCompressed) {
        m_atoi[ARG_SRC_3] = WEIGHT_SCALES;
        m_atoi[ARG_SRC_4] = WEIGHT_ZERO_POINTS;
    }
}

void GatherMatmul::initSupportedPrimitiveDescriptors() {
    const auto& srcTypes = getOriginalInputPrecisions();
    auto dstTypes = getOriginalOutputPrecisions();
    if (!fusedWith.empty()) {
        dstTypes = fusedWith.back()->getOriginalOutputPrecisions();
    }

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    auto makeSrcDesc = [&](size_t port) -> MemoryDescPtr {
        if (port >= srcTypes.size() || srcTypes[port] == ov::element::dynamic) {
            return MemoryDescUtils::makeEmptyDesc();
        }
        return creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcTypes[port], getInputShapeAtPort(port));
    };

    MemoryDescArgs descs;
    descs[ARG_SRC] = makeSrcDesc(DATA);
    descs[ARG_WEI] = makeSrcDesc(WEIGHTS);
    descs[ARG_SRC_1] = makeSrcDesc(INDICES);
    descs[ARG_BIAS] = makeSrcDesc(BIAS);
    descs[ARG_SRC_3] = m_isCompressed ? makeSrcDesc(WEIGHT_SCALES) : MemoryDescUtils::makeEmptyDesc();
    descs[ARG_SRC_4] = m_isCompressed ? makeSrcDesc(WEIGHT_ZERO_POINTS) : MemoryDescUtils::makeEmptyDesc();
    descs[ARG_DST] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dstTypes.front(), getOutputShapeAtPort(0));

    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    m_factory = std::make_shared<ExecutorFactory<GatherMatmulAttrs>>(m_attrs, executionContext, descs);

    const auto nodeDescriptorsList = m_factory->getProperMemoryDescriptors(descs);
    for (const auto& nodeDescriptors : nodeDescriptorsList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(srcTypes.size());

        for (const auto& [argId, portId] : m_atoi) {
            if (nodeDescriptors.count(argId)) {
                nodeConfig.inConfs[portId] = PortConfig{nodeDescriptors.at(argId)};
            }
        }

        nodeConfig.outConfs.emplace_back(nodeDescriptors.at(ARG_DST));

        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }
}

void GatherMatmul::createPrimitive() {
    for (const auto& [argId, portId] : m_atoi) {
        m_memory[argId] = getSrcMemoryAtPort(portId);
    }

    if (!m_isCompressed) {
        m_memory[ARG_SRC_3] = MemoryDescUtils::makeEmptyMemory(context);
        m_memory[ARG_SRC_4] = MemoryDescUtils::makeEmptyMemory(context);
    }
    m_memory[ARG_DST] = getDstMemoryAtPort(0);

    m_executor = m_factory->make(m_memory);

    Node::createPrimitive();

    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void GatherMatmul::prepareParams() {
    for (const auto& [argId, portId] : m_atoi) {
        m_memory[argId] = getSrcMemoryAtPort(portId);
    }
    m_memory[ARG_DST] = getDstMemoryAtPort(0);

    m_executor->update(m_memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

bool GatherMatmul::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);  // only data shape matters
}

void GatherMatmul::execute(const dnnl::stream& /*strm*/) {
    m_executor->execute(m_memory);
}

void GatherMatmul::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool GatherMatmul::created() const {
    return getType() == Type::GatherMatmul;
}

}  // namespace ov::intel_cpu::node
