// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grouped_matmul.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_map>

#include "common/blocked_desc_creator.h"
#include "config.h"
#include "cpu_types.h"
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#endif
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "node.h"
#include "node_config.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/grouped_matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "ov_ops/grouped_matmul_compressed.hpp"
#include "shape_inference/custom/grouped_matmul.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

namespace {

// The 2D x 3D form carries an offsets input, the 3D x 3D one does not. The rank of mat_a is the
// discriminator for both the public and the compressed op.
bool has_offsets(const ov::Node& op) {
    const auto& a_rank = op.get_input_partial_shape(0).rank();
    return a_rank.is_static() && a_rank.get_length() == 2;
}

// Index of the first optional input: the decompression scale of the compressed op.
size_t first_optional_port(const ov::Node& op) {
    return has_offsets(op) ? 3LU : 2LU;
}

}  // namespace

bool GroupedMatMul::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& error_message) noexcept {
    try {
        const bool is_compressed = ov::is_type<ov::op::internal::GroupedMatMulCompressed>(op);
        if (!is_compressed && !ov::is_type<ov::op::v17::GroupedMatMul>(op)) {
            error_message = "Only GroupedMatMul and GroupedMatMulCompressed operations are supported. Got: " +
                           std::string(op->get_type_info().name);
            return false;
        }

        const auto& a_shape = op->get_input_partial_shape(DATA);
        const auto& b_shape = op->get_input_partial_shape(WEIGHTS);
        if (a_shape.rank().is_dynamic() || !any_of(a_shape.rank().get_length(), 2, 3)) {
            error_message = "GroupedMatMul mat_a must be of a static rank 2 or 3";
            return false;
        }
        if (b_shape.rank().is_dynamic() || b_shape.rank().get_length() != 3) {
            error_message = "GroupedMatMul mat_b must be of a static rank 3";
            return false;
        }

        if (!ov::op::util::is_on_path<ov::op::v0::Constant>(op->input_value(WEIGHTS))) {
            error_message = "Only constant weights are supported for GroupedMatMul operation";
            return false;
        }

        if (is_compressed) {
            auto port = first_optional_port(*op);
            for (; port < op->get_input_size(); port++) {
                if (!ov::op::util::is_on_path<ov::op::v0::Constant>(op->input_value(port))) {
                    error_message =
                        "Only constant weight scales and zero points are supported for GroupedMatMul operation";
                    return false;
                }
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool GroupedMatMul::isSupportedCompressedOperation(const std::shared_ptr<ov::Node>& op,
                                                   [[maybe_unused]] size_t IC,
                                                   [[maybe_unused]] size_t OC,
                                                   [[maybe_unused]] size_t G,
                                                   [[maybe_unused]] const Config& config) noexcept {
#ifdef OPENVINO_ARCH_X86_64
    try {
        const auto& activations = getSupportedCompressedActivationsTypes();
        if (std::find(activations.begin(), activations.end(), op->get_input_element_type(0)) == activations.end()) {
            return false;
        }
    
        std::string error_message;
        if (!isSupportedOperation(op, error_message)) {
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

ov::element::TypeVector GroupedMatMul::getSupportedCompressedWeightsTypes() {
#ifdef OPENVINO_ARCH_X86_64
    using ov::element::Type_t;
    return {Type_t::u8, Type_t::i8, Type_t::u4, Type_t::i4};
#else
    return {};
#endif
}

ov::element::TypeVector GroupedMatMul::getSupportedCompressedActivationsTypes() {
    using ov::element::Type_t;
    // @todo enable for bf16 as well
    // after EnforceInferencePrecision is replaced with ConvertPrecision
    return {Type_t::f32};
}

bool GroupedMatMul::isSupportedOperation(const std::shared_ptr<const ov::Node>& op) noexcept {
    std::string error_message;
    return isSupportedOperation(op, error_message);
}

GroupedMatMul::GroupedMatMul(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, GroupedMatMulShapeInferFactory()) {
    std::string error_message;
    if (!isSupportedOperation(op, error_message)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(error_message);
    }

    algorithm = ov::is_type<ov::op::internal::GroupedMatMulCompressed>(op) ? Algorithm::GroupedMatMulCompressed
                                                                           : Algorithm::GroupedMatMulDefault;

    m_atoi[ARG_SRC] = DATA;
    m_atoi[ARG_WEI] = WEIGHTS;
    if (has_offsets(*op)) {
        m_atoi[ARG_SRC_1] = OFFSETS;
    }
    if (algorithm == Algorithm::GroupedMatMulCompressed) {
        const auto scales_port = first_optional_port(*op);
        m_atoi[ARG_SRC_3] = static_cast<int>(scales_port);
        if (op->get_input_size() > scales_port + 1) {
            m_atoi[ARG_SRC_4] = static_cast<int>(scales_port + 1);
        }
    }
}

void GroupedMatMul::initSupportedPrimitiveDescriptors() {
    const auto& src_types = getOriginalInputPrecisions();
    auto dst_types = getOriginalOutputPrecisions();
    if (!fusedWith.empty()) {
        dst_types = fusedWith.back()->getOriginalOutputPrecisions();
    }

    const auto& creators_map = BlockedDescCreator::getCommonCreators();

    auto make_src_desc = [&](size_t port) -> MemoryDescPtr {
        if (port >= src_types.size() || src_types[port] == ov::element::dynamic) {
            return MemoryDescUtils::makeEmptyDesc();
        }
        return creators_map.at(LayoutType::ncsp)->createSharedDesc(src_types[port], getInputShapeAtPort(port));
    };

    MemoryDescArgs descs;
    for (const auto& [arg_id, port_id] : m_atoi) {
        descs[arg_id] = make_src_desc(port_id);
    }
    // GroupedMatMul-17 has no bias; the empty slot is what the executor's make_bias_md() expects
    descs[ARG_BIAS] = MemoryDescUtils::makeEmptyDesc();
    for (auto arg_id : {ARG_SRC_1, ARG_SRC_3, ARG_SRC_4}) {
        if (descs.count(arg_id) == 0) {
            descs[arg_id] = MemoryDescUtils::makeEmptyDesc();
        }
    }
    descs[ARG_DST] = creators_map.at(LayoutType::ncsp)->createSharedDesc(dst_types.front(), getOutputShapeAtPort(0));

    auto execution_context = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    m_factory = std::make_shared<ExecutorFactory<GroupedMatMulAttrs>>(m_attrs, execution_context, descs);

    const auto node_descriptors_list = m_factory->getProperMemoryDescriptors(descs);
    for (const auto& node_descriptors : node_descriptors_list) {
        NodeConfig node_config;
        node_config.inConfs.resize(src_types.size());

        for (const auto& [arg_id, port_id] : m_atoi) {
            if (node_descriptors.count(arg_id)) {
                node_config.inConfs[port_id] = PortConfig{node_descriptors.at(arg_id)};
            }
        }

        node_config.outConfs.emplace_back(node_descriptors.at(ARG_DST));

        supportedPrimitiveDescriptors.emplace_back(node_config, impl_desc_type::undef);
    }
}

void GroupedMatMul::createPrimitive() {
    for (const auto& [arg_id, port_id] : m_atoi) {
        m_memory[arg_id] = getSrcMemoryAtPort(port_id);
    }

    for (auto arg_id : {ARG_BIAS, ARG_SRC_1, ARG_SRC_3, ARG_SRC_4}) {
        if (m_memory.count(arg_id) == 0) {
            m_memory[arg_id] = MemoryDescUtils::makeEmptyMemory(context);
        }
    }
    m_memory[ARG_DST] = getDstMemoryAtPort(0);

    m_executor = m_factory->make(m_memory);

    Node::createPrimitive();

    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void GroupedMatMul::prepareParams() {
    for (const auto& [arg_id, port_id] : m_atoi) {
        m_memory[arg_id] = getSrcMemoryAtPort(port_id);
    }
    m_memory[ARG_DST] = getDstMemoryAtPort(0);

    m_executor->update(m_memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

bool GroupedMatMul::isExecutable() const {
    return !isInputTensorAtPortEmpty(DATA);  // only data shape matters
}

void GroupedMatMul::execute(const dnnl::stream& /*strm*/) {
    m_executor->execute(m_memory);
}

void GroupedMatMul::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool GroupedMatMul::created() const {
    return getType() == Type::GroupedMatMul;
}

}  // namespace ov::intel_cpu::node
