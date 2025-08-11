// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/matrix_nms.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
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
#include "snippets/op/brgemm.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/broadcastmove.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/convert_truncation.hpp"
#include "snippets/op/fill.hpp"
#include "snippets/op/horizon_max.hpp"
#include "snippets/op/horizon_sum.hpp"
#include "snippets/op/kernel.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/nop.hpp"
#include "snippets/op/perf_count.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/op/reshape.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/op/store.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/op/vector_buffer.hpp"
#include "transformations/cpu_opset/common/op/causal_mask_preprocess.hpp"
#include "transformations/cpu_opset/common/op/leaky_relu.hpp"
#include "transformations/cpu_opset/common/op/ngram.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"
#include "transformations/cpu_opset/common/op/read_value_with_subgraph.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#if defined(OPENVINO_ARCH_X86_64)
#    include "transformations/cpu_opset/x64/op/interaction.hpp"
#    include "transformations/cpu_opset/x64/op/llm_mlp.hpp"
#    include "transformations/cpu_opset/x64/op/qkv_proj.hpp"
#    include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#    include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#    include "transformations/snippets/x64/op/load_convert.hpp"
#    include "transformations/snippets/x64/op/perf_count_rdtsc.hpp"
#    include "transformations/snippets/x64/op/store_convert.hpp"
#elif defined(OPENVINO_ARCH_ARM64)
#    include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#    include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#endif

namespace {

template <typename Op>
class TypeRelaxedExtension : public ov::OpExtension<ov::op::TypeRelaxed<Op>> {
public:
    TypeRelaxedExtension() : m_ext_type(Op::get_type_info_static().name, version()) {}

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
    static const char* version() {
        static const auto version =
            std::string(ov::op::TypeRelaxedBase::version_prefix) + Op::get_type_info_static().version_id;
        return version.data();
    }

    ov::DiscreteTypeInfo m_ext_type;
};

}  // namespace

#if defined(OPENVINO_ARCH_X86_64)
#    define OP_EXTENSION_X64(x) x,
#else
#    define OP_EXTENSION_X64(x)
#endif

#if defined(OPENVINO_ARCH_ARM64)
#    define OP_EXTENSION_ARM64(x) x,
#else
#    define OP_EXTENSION_ARM64(x)
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
    std::make_shared<ov::OpExtension<ov::op::PagedAttentionExtension>>(),
    // clang-format off
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
    OP_EXTENSION_ARM64(std::make_shared<ov::OpExtension<ov::intel_cpu::aarch64::GemmCPU>>())
    OP_EXTENSION_ARM64(std::make_shared<ov::OpExtension<ov::intel_cpu::aarch64::GemmCopyB>>())
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
