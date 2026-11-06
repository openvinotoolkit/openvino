// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_residual.h"

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <set>
#include <string>
#include <vector>

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#endif
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/gated_residual_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gated_residual.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

namespace {

using ReadInputIdx = ov::op::internal::GatedResidual::ReadInputIdx;
using WriteInputIdx = ov::op::internal::GatedResidual::WriteInputIdx;

}  // namespace

GatedResidual::GatedResidual(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto gr = ov::as_type_ptr<const ov::op::internal::GatedResidual>(op);
    OPENVINO_ASSERT(gr, "Expected GatedResidual node");
    m_attrs.mode = gr->get_mode();
    m_attrs.hc_count = gr->get_hc_count();
    m_attrs.hidden_size = gr->get_hidden_size();
    m_attrs.lowrank = gr->get_lowrank();
    m_attrs.epsilon = gr->get_epsilon();
    m_attrs.norm_gain_plus_one = gr->get_norm_gain_plus_one();
    m_attrs.produce_write_gate = gr->get_produce_write_gate();
    if (m_attrs.mode != ov::op::internal::GatedResidual::Mode::WRITE) {
        const auto layout = gr->get_read_layout();
        OPENVINO_ASSERT(layout.has_value(), "GatedResidual (READ/FINAL_MIX) has an unexpected number of inputs");
        m_attrs.read_layout = *layout;
    }
}

void GatedResidual::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // Negotiate a single data precision for this node instance from the main data input (RESIDUAL
    // is port 0 in both ReadInputIdx and WriteInputIdx) and apply it uniformly to every data-path
    // port, falling back to f32 for anything else the graph might otherwise propose. Internal
    // accumulation always happens in f32 regardless.
    auto precision = getOriginalInputPrecisionAtPort(0);
    if (none_of(precision, ov::element::f32, ov::element::bf16, ov::element::f16)) {
        precision = ov::element::f32;
    }

    // Compressed weight ports (u8/i8/u4/i4), their floating-point scale ports and their zero-point ports (the weights'
    // type) must keep their own native element type here. Forcing them to the data `precision` would make the graph
    // insert an implicit value-converting Reorder on that edge -- e.g. turning a u8 weight buffer into f32, where each
    // raw byte value v becomes the 4-byte encoding of v.0f -- silently corrupting the data the executor reads back as
    // raw bytes.
    const auto& layout = m_attrs.read_layout;
    std::set<size_t> native_precision_ports;
    if (m_attrs.mode != ov::op::internal::GatedResidual::Mode::WRITE && layout.compressed) {
        native_precision_ports = {ReadInputIdx::W_DOWN,
                                  ReadInputIdx::SCALE_DOWN,
                                  ReadInputIdx::W_UP,
                                  ReadInputIdx::SCALE_UP};
        if (layout.has_zp) {
            native_precision_ports.insert(ReadInputIdx::ZP_DOWN);
            native_precision_ports.insert(ReadInputIdx::ZP_UP);
        }
        if (layout.write_gate) {
            native_precision_ports.insert(layout.w_inject);
            native_precision_ports.insert(layout.scale_inject);
            if (layout.has_zp) {
                native_precision_ports.insert(layout.zp_inject);
            }
        }
    }

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    MemoryDescArgs descs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        const auto prc = native_precision_ports.count(i) != 0U ? getOriginalInputPrecisionAtPort(i) : precision;
        descs[gatedResidualSrcArg(i)] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prc, getInputShapeAtPort(i));
    }
    for (size_t i = 0; i < getOriginalOutputsNumber(); ++i) {
        descs[gatedResidualDstArg(i)] =
            creatorsMap.at(LayoutType::ncsp)->createSharedDesc(precision, getOutputShapeAtPort(i));
    }

    if (m_attrs.mode != ov::op::internal::GatedResidual::Mode::WRITE) {
        // The projection weights are the fused GEMM weights and, for compressed weights, their decompression scales
        // and zero-points: all graph constants. The executor prepares them once (prepacked weights, converted
        // decompression views), so non-constant inputs are not supported. Constant status is known here (it is
        // updated as graph edges are added).
        std::vector<size_t> constant_ports{ReadInputIdx::W_DOWN, ReadInputIdx::W_UP};
        if (layout.write_gate) {
            constant_ports.push_back(layout.w_inject);
        }
        if (layout.compressed) {
            constant_ports.push_back(ReadInputIdx::SCALE_DOWN);
            constant_ports.push_back(ReadInputIdx::SCALE_UP);
            if (layout.has_zp) {
                constant_ports.push_back(ReadInputIdx::ZP_DOWN);
                constant_ports.push_back(ReadInputIdx::ZP_UP);
            }
            if (layout.write_gate) {
                constant_ports.push_back(layout.scale_inject);
                if (layout.has_zp) {
                    constant_ports.push_back(layout.zp_inject);
                }
            }
        }
        for (const auto port : constant_ports) {
            if (!getParentEdgeAt(port)->getParent()->isConstant()) {
                OPENVINO_THROW_NOT_IMPLEMENTED("GatedResidual requires constant weights and decompression scales/"
                                               "zero-points (input ",
                                               port,
                                               " is not constant).");
            }
        }
    }

    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    m_factory = std::make_shared<ExecutorFactory<GatedResidualAttrs>>(m_attrs, executionContext, descs);

    const auto nodeDescriptorsList = m_factory->getProperMemoryDescriptors(descs);
    for (const auto& nodeDescriptors : nodeDescriptorsList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(getParentEdges().size());
        for (size_t i = 0; i < getParentEdges().size(); ++i) {
            nodeConfig.inConfs[i] = PortConfig{nodeDescriptors.at(gatedResidualSrcArg(i))};
        }
        for (size_t i = 0; i < getOriginalOutputsNumber(); ++i) {
            nodeConfig.outConfs.emplace_back(nodeDescriptors.at(gatedResidualDstArg(i)));
        }
        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }
}

void GatedResidual::updateMemoryArgs() {
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        m_memory[gatedResidualSrcArg(i)] = getSrcMemoryAtPort(i);
    }
    for (size_t i = 0; i < getOriginalOutputsNumber(); ++i) {
        m_memory[gatedResidualDstArg(i)] = getDstMemoryAtPort(i);
    }
}

void GatedResidual::createPrimitive() {
    updateMemoryArgs();
    m_executor = m_factory->make(m_memory);

    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void GatedResidual::prepareParams() {
    updateMemoryArgs();
    OPENVINO_ASSERT(m_executor->update(m_memory), "GatedResidual executor update failed");
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void GatedResidual::execute([[maybe_unused]] const dnnl::stream& strm) {
    updateMemoryArgs();
    m_executor->execute(m_memory);
}

bool GatedResidual::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
    try {
        const auto gr = ov::as_type_ptr<const ov::op::internal::GatedResidual>(op);
        if (!gr) {
            errorMessage = "Only GatedResidual operation is supported";
            return false;
        }
#if !defined(OPENVINO_ARCH_X86_64)
        // The only executor is the oneDNN/JIT one, which exists on x86-64 only.
        errorMessage = "GatedResidual is only supported on x86-64.";
        return false;
#endif

        const auto mode = gr->get_mode();
        std::set<size_t> weight_ports;
        std::set<size_t> scale_ports;
        std::set<size_t> zero_point_ports;
        if (mode != ov::op::internal::GatedResidual::Mode::WRITE) {
            const auto layout = gr->get_read_layout();
            if (!layout.has_value()) {
                errorMessage = "GatedResidual has an unexpected number of inputs.";
                return false;
            }
            if (layout->compressed) {
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
                if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::cpu_isa_t::avx2)) {
                    errorMessage = "GatedResidual compressed weights require at least AVX2.";
                    return false;
                }
#else
                errorMessage = "GatedResidual compressed weights are only supported on x86-64 with AVX2.";
                return false;
#endif
                weight_ports = {ReadInputIdx::W_DOWN, ReadInputIdx::W_UP};
                scale_ports = {ReadInputIdx::SCALE_DOWN, ReadInputIdx::SCALE_UP};
                if (layout->has_zp) {
                    zero_point_ports = {ReadInputIdx::ZP_DOWN, ReadInputIdx::ZP_UP};
                }
                if (layout->write_gate) {
                    weight_ports.insert(layout->w_inject);
                    scale_ports.insert(layout->scale_inject);
                    if (layout->has_zp) {
                        zero_point_ports.insert(layout->zp_inject);
                    }
                }

                // Per projection the scale is [G, N] and the group size along K is K / G; the op guarantees that G
                // divides K, so any static group count (1 == per-channel, 2, 3, ...) is valid.
                const auto groups_static = [&](size_t scale_port) {
                    const auto& ps = op->get_input_partial_shape(scale_port);
                    return ps.rank().is_static() && ps[0].is_static();
                };
                if (!groups_static(ReadInputIdx::SCALE_DOWN) || !groups_static(ReadInputIdx::SCALE_UP) ||
                    (layout->write_gate && !groups_static(layout->scale_inject))) {
                    errorMessage = "GatedResidual compressed weight scales must have a static group count.";
                    return false;
                }
            }
        }

        for (size_t i = 0; i < op->get_input_size(); i++) {
            const auto et = op->get_input_element_type(i);
            if (weight_ports.count(i) != 0U) {
                if (none_of(et, ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4)) {
                    errorMessage = "GatedResidual CPU implementation only supports u8/i8/u4/i4 compressed weights.";
                    return false;
                }
                continue;
            }
            if (scale_ports.count(i) != 0U) {
                if (none_of(et, ov::element::f32, ov::element::f16, ov::element::bf16)) {
                    errorMessage = "GatedResidual CPU implementation only supports f32/f16/bf16 weight decompression "
                                   "scales.";
                    return false;
                }
                continue;
            }
            if (zero_point_ports.count(i) != 0U) {
                if (none_of(et, ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4)) {
                    errorMessage = "GatedResidual CPU implementation only supports u8/i8/u4/i4 weight decompression "
                                   "zero-points.";
                    return false;
                }
                continue;
            }
            if (none_of(et, ov::element::f32, ov::element::bf16, ov::element::f16)) {
                errorMessage = "GatedResidual CPU implementation only supports f32/bf16/f16 inputs.";
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
