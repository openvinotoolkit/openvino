// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_residual.h"

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "cpu_memory.h"
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
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

namespace {

using Mode = ov::op::internal::GatedResidual::Mode;

}  // namespace

GatedResidual::GatedResidual(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto gr = ov::as_type_ptr<const ov::op::internal::GatedResidual>(op);
    CPU_NODE_ASSERT(gr, "expected GatedResidual node");
    const auto& config = gr->get_config();
    m_attrs.hc_count = config.hc_count;
    m_attrs.hidden_size = config.hidden_size;
    m_attrs.lowrank = config.lowrank;
    m_attrs.epsilon = config.epsilon;
    m_attrs.produce_write_gate = config.produce_write_gate;
    const auto layout = gr->get_read_layout();
    CPU_NODE_ASSERT(layout.has_value(), "has an unexpected number of inputs");
    m_attrs.read_layout = *layout;
}

void GatedResidual::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // Every port keeps its original precision; the executor's type mapping picks the ones it runs in.
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    MemoryDescArgs descs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        descs[gatedResidualSrcArg(i)] =
            creatorsMap.at(LayoutType::ncsp)
                ->createSharedDesc(getOriginalInputPrecisionAtPort(i), getInputShapeAtPort(i));
    }
    for (size_t i = 0; i < getOriginalOutputsNumber(); ++i) {
        descs[gatedResidualDstArg(i)] =
            creatorsMap.at(LayoutType::ncsp)
                ->createSharedDesc(getOriginalOutputPrecisionAtPort(i), getOutputShapeAtPort(i));
    }

    // Each projection gets its own context: the private weight cache is keyed by the packed layout only. The
    // projections run one after another, so they share one scratchpad, which the executor's other buffers do not use.
    const auto projectionPad = context->getScratchPad()->child(kPadProjections);
    for (auto& projection_context : m_attrs.projection_contexts) {
        projection_context =
            std::make_shared<ExecutorContext>(context,
                                              getImplPriority(),
                                              std::make_shared<std::unordered_map<std::string, MemoryPtr>>(),
                                              projectionPad);
    }

    m_attrs.dynamic_quantization_group_size = context->getConfig().fcDynamicQuantizationGroupSize;
    m_attrs.model_type = context->getConfig().modelType;

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

const std::vector<impl_desc_type>& GatedResidual::getDefaultImplPriority() {
    // The projections run through FC, so use the FC node's brgemm-first priority. `undef` is this node's own descriptor
    // type.
    static const std::vector<impl_desc_type> priorities = {
        impl_desc_type::unknown,
        impl_desc_type::undef,
        impl_desc_type::brgemm_avx512_amx,
        impl_desc_type::brgemm_avx512,
        impl_desc_type::brgemm_avx2,
        impl_desc_type::gemm_blas,
        impl_desc_type::gemm_avx512,
        impl_desc_type::gemm_avx2,
        impl_desc_type::gemm_avx,
        impl_desc_type::gemm_any,
        impl_desc_type::gemm,
        impl_desc_type::jit_gemm,
        impl_desc_type::ref,
    };

    return priorities;
}

void GatedResidual::bindMemoryArgs() {
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        m_memory[gatedResidualSrcArg(i)] = getSrcMemoryAtPort(i);
    }
    for (size_t i = 0; i < getOriginalOutputsNumber(); ++i) {
        m_memory[gatedResidualDstArg(i)] = getDstMemoryAtPort(i);
    }
}

void GatedResidual::createPrimitive() {
    // Edge memories do not change after Graph::Allocate() (reshapes update descriptors in place, user tensors swap the
    // block), so bind once.
    bindMemoryArgs();
    m_executor = m_factory->make(m_memory);

    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void GatedResidual::prepareParams() {
    CPU_NODE_ASSERT(m_executor->update(m_memory), "executor update failed");
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void GatedResidual::execute([[maybe_unused]] const dnnl::stream& strm) {
    m_executor->execute(m_memory);
}

ov::element::TypeVector GatedResidual::getSupportedCompressedWeightsTypes() {
    return {ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4};
}

bool GatedResidual::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
    try {
        const auto gr = ov::as_type_ptr<const ov::op::internal::GatedResidual>(op);
        if (!gr) {
            errorMessage = "Only GatedResidual operation is supported";
            return false;
        }
        // Every mode runs JIT kernels built for avx2 and avx512_core only.
        if (!ov::with_cpu_x86_avx2()) {
            errorMessage = "GatedResidual requires at least AVX2.";
            return false;
        }

        // WRITE is not executed here: GatedResidualWriteSinkReshape turns that pattern into one Eltwise.
        if (gr->get_config().mode == Mode::WRITE) {
            errorMessage = "GatedResidual WRITE mode is not supported; the CPU plugin runs the pattern as Eltwise.";
            return false;
        }

        std::set<size_t> weight_ports;
        std::set<size_t> scale_ports;
        std::set<size_t> zero_point_ports;
        const auto layout = gr->get_read_layout();
        if (!layout.has_value()) {
            errorMessage = "GatedResidual has an unexpected number of inputs.";
            return false;
        }
        if (layout->compressed) {
            weight_ports = {layout->w_down, layout->w_up};
            scale_ports = {layout->scale_down, layout->scale_up};
            if (layout->has_zp) {
                zero_point_ports = {layout->zp_down, layout->zp_up};
            }
            if (layout->write_gate) {
                weight_ports.insert(layout->w_inject);
                scale_ports.insert(layout->scale_inject);
                if (layout->has_zp) {
                    zero_point_ports.insert(layout->zp_inject);
                }
            }

            // The scale is [N, G] or one element (one group); the op guarantees that G divides K, so any static group
            // count is valid.
            const auto groups_static = [&](size_t scale_port) {
                const auto& ps = op->get_input_partial_shape(scale_port);
                return ps.is_static() && (ov::shape_size(ps.to_shape()) == 1 || ps.size() == 2);
            };
            if (!groups_static(layout->scale_down) || !groups_static(layout->scale_up) ||
                (layout->write_gate && !groups_static(layout->scale_inject))) {
                errorMessage = "GatedResidual compressed weight scales must have a static group count.";
                return false;
            }
        }

        // Weights, scales and zero-points are prepared once by the executor, so they must be constants, or a subgraph
        // of constants (the decompression of weights before ConvertGatedResidualToCompressed, folded afterwards).
        const auto is_constant_input = [&](size_t i) {
            return ov::op::util::is_on_path<ov::op::v0::Constant>(op->input_value(i));
        };
        std::set<size_t> constant_ports;
        constant_ports.insert(weight_ports.begin(), weight_ports.end());
        constant_ports.insert(scale_ports.begin(), scale_ports.end());
        constant_ports.insert(zero_point_ports.begin(), zero_point_ports.end());
        // Uncompressed weights are not in weight_ports (it only drives the compressed type check).
        constant_ports.insert({layout->w_down, layout->w_up});
        if (layout->write_gate) {
            constant_ports.insert(layout->w_inject);
        }
        for (const auto port : constant_ports) {
            if (!is_constant_input(port)) {
                errorMessage = "GatedResidual requires constant weights and decompression scales/zero-points.";
                return false;
            }
        }

        for (size_t i = 0; i < op->get_input_size(); i++) {
            const auto et = op->get_input_element_type(i);
            if (weight_ports.count(i) != 0U) {
                if (!contains(getSupportedCompressedWeightsTypes(), et)) {
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
                if (!contains(getSupportedCompressedWeightsTypes(), et)) {
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

bool GatedResidual::isSupportedCompressedOperation(const std::shared_ptr<const ov::Node>& op) noexcept {
    try {
        std::string errorMessage;
        if (!isSupportedOperation(op, errorMessage)) {
            return false;
        }
        const auto gr = ov::as_type_ptr<const ov::op::internal::GatedResidual>(op);
        const auto layout = gr ? gr->get_read_layout() : std::nullopt;
        if (!layout || !layout->compressed) {
            return false;
        }
        // The limits of FC weights decompression, which the projections run through: groups of at least 4 elements
        // along K, and more than one output channel.
        const auto supported = [&](size_t weights_port, size_t scale_port) {
            const auto& weights_shape = op->get_input_shape(weights_port);
            const size_t N = weights_shape[0];
            const size_t K = weights_shape[1];
            const auto& scale_shape = op->get_input_shape(scale_port);
            const size_t G = ov::shape_size(scale_shape) == 1 ? 1 : scale_shape[1];
            return G != 0 && K % G == 0 && K / G >= 4 && N != 1;
        };
        return supported(layout->w_down, layout->scale_down) && supported(layout->w_up, layout->scale_up) &&
               (!layout->write_gate || supported(layout->w_inject, layout->scale_inject));
    } catch (...) {
        return false;
    }
}

}  // namespace ov::intel_cpu::node
