// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include "openvino/core/op_extension.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "ov_ops/fully_connected_quantized.hpp"
#include "ov_ops/fully_connected_quantized_legacy.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "snippets/op/subgraph.hpp"
#include "transformations/cpu_opset/common/op/causal_mask_preprocess.hpp"
#include "transformations/cpu_opset/common/op/leaky_relu.hpp"
#include "transformations/cpu_opset/common/op/ngram.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"
#include "transformations/cpu_opset/common/op/read_value_with_subgraph.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/cpu_opset/x64/op/interaction.hpp"
#include "transformations/cpu_opset/x64/op/llm_mlp.hpp"
#include "transformations/cpu_opset/x64/op/mha.hpp"
#include "transformations/cpu_opset/x64/op/qkv_proj.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/perf_count_rdtsc.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"

namespace {

template <typename Op>
class TypeRelaxedExtension : public ov::OpExtension<ov::op::TypeRelaxed<Op>> {
public:
    TypeRelaxedExtension() : m_ext_type(Op::get_type_info_static().name, "type_relaxed_opset") {}
    ~TypeRelaxedExtension() override = default;

    [[nodiscard]] const ov::DiscreteTypeInfo& get_type_info() const override {
        return m_ext_type;
    }

    ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) const override {
        return ov::OpExtension<ov::op::TypeRelaxed<Op>>::create(inputs, visitor);
    }

    [[nodiscard]] std::vector<ov::Extension::Ptr> get_attached_extensions() const override {
        return {};
    }

private:
    ov::DiscreteTypeInfo m_ext_type;
};

}  // namespace

#if defined(OPENVINO_ARCH_X86_64)
#    define OP_EXTENSION_X64(x) x,
#else
#    define OP_EXTENSION_X64(x)
#endif

#if defined(SNIPPETS_DEBUG_CAPS)
#    define OP_EXTENSION_SNIPPETS_DEBUG_CAPS(x) x,
#else
#    define OP_EXTENSION_SNIPPETS_DEBUG_CAPS(x)
#endif

#if defined(SNIPPETS_DEBUG_CAPS) && defined(OPENVINO_ARCH_X86_64)
#    define OP_EXTENSION_SNIPPETS_DEBUG_CAPS_X64(x) x,
#else
#    define OP_EXTENSION_SNIPPETS_DEBUG_CAPS_X64(x)
#endif

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    // CPU extensions
    std::make_shared<ov::OpExtension<ov::intel_cpu::LeakyReluNode>>(),
    std::make_shared<ov::OpExtension<ov::intel_cpu::PowerStaticNode>>(),
    std::make_shared<ov::OpExtension<ov::intel_cpu::CausalMaskPreprocessNode>>(),
    std::make_shared<ov::OpExtension<ov::intel_cpu::SwishNode>>(),
    std::make_shared<ov::OpExtension<ov::intel_cpu::SDPAWithTransposeReshape>>(),
    std::make_shared<ov::OpExtension<ov::intel_cpu::NgramNode>>(),
    std::make_shared<ov::OpExtension<ov::intel_cpu::ReadValueWithSubgraph>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::GatherCompressed>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::NonMaxSuppressionIEInternal>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::MulticlassNmsIEInternal>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::AUGRUCell>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::AUGRUSequence>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::NmsStaticShapeIE<ov::op::v8::MatrixNms>>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::RMS>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::RoPE>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::FullyConnected>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::FullyConnectedCompressed>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::FullyConnectedQuantizedLegacy>>(),
    std::make_shared<ov::OpExtension<ov::op::internal::FullyConnectedQuantized>>(),
    // clang-format off
    OP_EXTENSION_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::MHANode>>())
    OP_EXTENSION_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::InteractionNode>>())
    OP_EXTENSION_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::LLMMLPNode>>())
    OP_EXTENSION_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::QKVProjectionNode>>())
    OP_EXTENSION_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>>())
    OP_EXTENSION_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::LoadConvertSaturation>>())
    OP_EXTENSION_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::LoadConvertTruncation>>())
    OP_EXTENSION_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::StoreConvertSaturation>>())
    OP_EXTENSION_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::StoreConvertTruncation>>())
    OP_EXTENSION_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::BrgemmCPU>>())
    OP_EXTENSION_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::BrgemmCopyB>>())
    // clang-format on
    std::make_shared<TypeRelaxedExtension<ov::op::v1::Add>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::AvgPool>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v14::AvgPool>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::Clamp>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::Concat>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::Convolution>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::ConvolutionBackpropData>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::DepthToSpace>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::Equal>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::FakeQuantize>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::Greater>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::GreaterEqual>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::GroupConvolution>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::GroupConvolutionBackpropData>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::Interpolate>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v4::Interpolate>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::Less>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::LessEqual>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::LogicalAnd>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::LogicalNot>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::LogicalOr>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::LogicalXor>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::MatMul>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::MaxPool>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::Multiply>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::NormalizeL2>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::NotEqual>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::PRelu>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::Relu>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::ReduceMax>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::ReduceLogicalAnd>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::ReduceLogicalOr>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::ReduceMean>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::ReduceMin>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::ReduceSum>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::Reshape>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::Select>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::ShapeOf>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::ShuffleChannels>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::Squeeze>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v1::Subtract>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::Unsqueeze>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v0::MVN>>(),
    std::make_shared<TypeRelaxedExtension<ov::op::v6::MVN>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::Brgemm>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::BroadcastLoad>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::BroadcastMove>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::ConvertSaturation>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::ConvertTruncation>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::Fill>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::HorizonMax>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::HorizonSum>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::KernelStatic>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::KernelDynamic>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::Load>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::LoadReorder>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::LoopBegin>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::LoopEnd>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::Buffer>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::Nop>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::PowerStatic>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::Scalar>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::Store>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::Subgraph>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::VectorBuffer>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::RankNormalization>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::ReduceMax>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::ReduceSum>>(),
    std::make_shared<ov::OpExtension<ov::snippets::op::Reshape>>(),
    // clang-format off
    OP_EXTENSION_SNIPPETS_DEBUG_CAPS(std::make_shared<ov::OpExtension<ov::snippets::op::PerfCountBegin>>())
    OP_EXTENSION_SNIPPETS_DEBUG_CAPS(std::make_shared<ov::OpExtension<ov::snippets::op::PerfCountEnd>>())
    OP_EXTENSION_SNIPPETS_DEBUG_CAPS_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::PerfCountRdtscBegin>>())
    OP_EXTENSION_SNIPPETS_DEBUG_CAPS_X64(std::make_shared<ov::OpExtension<ov::intel_cpu::PerfCountRdtscEnd>>())
    // clang-format on
}));
