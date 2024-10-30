// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include "openvino/core/op_extension.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "snippets/op/subgraph.hpp"
#include "transformations/cpu_opset/common/op/causal_mask_preprocess.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "transformations/cpu_opset/common/op/leaky_relu.hpp"
#include "transformations/cpu_opset/common/op/ngram.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/cpu_opset/x64/op/interaction.hpp"
#include "transformations/cpu_opset/x64/op/mha.hpp"
#include "transformations/cpu_opset/x64/op/llm_mlp.hpp"
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
    TypeRelaxedExtension()
        : m_ext_type(Op::get_type_info_static().name, "type_relaxed_opset") {}
    ~TypeRelaxedExtension() override = default;

    const ov::DiscreteTypeInfo& get_type_info() const override {
        return m_ext_type;
    }

    ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) const override {
        return ov::OpExtension<ov::op::TypeRelaxed<Op>>::create(inputs, visitor);
    }

    std::vector<ov::Extension::Ptr> get_attached_extensions() const override {
        return {};
    }

private:
    ov::DiscreteTypeInfo m_ext_type;
};

}  // namespace

#define OP_EXTENSION(NAME) std::make_shared<ov::OpExtension<NAME>>(),

#define TYPE_RELAXED_OP_EXTENSION(NAME) std::make_shared<TypeRelaxedExtension<NAME>>(),

#if defined(OPENVINO_ARCH_X86_64)
#    define OP_EXTENSION_X64(NAME) OP_EXTENSION(NAME)
#else
#    define OP_EXTENSION_X64(NAME)
#endif

#define CPU_EXTENSIONS                                                      \
    OP_EXTENSION(ov::intel_cpu::FullyConnectedNode)                         \
    OP_EXTENSION(ov::intel_cpu::LeakyReluNode)                              \
    OP_EXTENSION(ov::intel_cpu::PowerStaticNode)                            \
    OP_EXTENSION(ov::intel_cpu::CausalMaskPreprocessNode)                   \
    OP_EXTENSION(ov::intel_cpu::SwishNode)                                  \
    OP_EXTENSION(ov::intel_cpu::SDPAWithTransposeReshape)                   \
    OP_EXTENSION(ov::intel_cpu::NgramNode)                                  \
    OP_EXTENSION(ov::op::internal::GatherCompressed)                        \
    OP_EXTENSION(ov::op::internal::NonMaxSuppressionIEInternal)             \
    OP_EXTENSION(ov::op::internal::MulticlassNmsIEInternal)                 \
    OP_EXTENSION(ov::op::internal::AUGRUCell)                               \
    OP_EXTENSION(ov::op::internal::AUGRUSequence)                           \
    OP_EXTENSION(ov::op::internal::NmsStaticShapeIE<ov::op::v8::MatrixNms>) \
    OP_EXTENSION(ov::op::internal::RMS)                                     \
    OP_EXTENSION(ov::op::internal::RoPE)                                    \
    OP_EXTENSION_X64(ov::intel_cpu::MHANode)                                \
    OP_EXTENSION_X64(ov::intel_cpu::InteractionNode)                        \
    OP_EXTENSION_X64(ov::intel_cpu::LLMMLPNode)                             \
    OP_EXTENSION_X64(ov::intel_cpu::QKVProjectionNode)                      \
    OP_EXTENSION_X64(ov::intel_cpu::ScaledDotProductAttentionWithKVCache)   \
    OP_EXTENSION_X64(ov::intel_cpu::LoadConvertSaturation)                  \
    OP_EXTENSION_X64(ov::intel_cpu::LoadConvertTruncation)                  \
    OP_EXTENSION_X64(ov::intel_cpu::StoreConvertSaturation)                 \
    OP_EXTENSION_X64(ov::intel_cpu::StoreConvertTruncation)                 \
    OP_EXTENSION_X64(ov::intel_cpu::BrgemmCPU)                              \
    OP_EXTENSION_X64(ov::intel_cpu::BrgemmCopyB)

#define TYPE_RELAXED_EXTENSIONS                                         \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::Add)                          \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::AvgPool)                      \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v14::AvgPool)                     \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::Clamp)                        \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::Concat)                       \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::Convolution)                  \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::ConvolutionBackpropData)      \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::DepthToSpace)                 \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::Equal)                        \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::FakeQuantize)                 \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::Greater)                      \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::GreaterEqual)                 \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::GroupConvolution)             \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::GroupConvolutionBackpropData) \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::Interpolate)                  \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v4::Interpolate)                  \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::Less)                         \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::LessEqual)                    \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::LogicalAnd)                   \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::LogicalNot)                   \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::LogicalOr)                    \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::LogicalXor)                   \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::MatMul)                       \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::MaxPool)                      \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::Multiply)                     \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::NormalizeL2)                  \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::NotEqual)                     \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::PRelu)                        \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::Relu)                         \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::ReduceMax)                    \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::ReduceLogicalAnd)             \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::ReduceLogicalOr)              \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::ReduceMean)                   \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::ReduceMin)                    \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::ReduceSum)                    \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::Reshape)                      \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::Select)                       \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::ShapeOf)                      \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::ShuffleChannels)              \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::Squeeze)                      \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v1::Subtract)                     \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::Unsqueeze)                    \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v0::MVN)                          \
    TYPE_RELAXED_OP_EXTENSION(ov::op::v6::MVN)

#ifdef SNIPPETS_DEBUG_CAPS
#    define SNIPPETS_DEBUG_CAPS_EXTENSIONS                   \
        OP_EXTENSION(ov::snippets::op::PerfCountBegin)       \
        OP_EXTENSION(ov::snippets::op::PerfCountEnd)         \
        OP_EXTENSION_X64(ov::intel_cpu::PerfCountRdtscBegin) \
        OP_EXTENSION_X64(ov::intel_cpu::PerfCountRdtscEnd)
#else
#    define SNIPPETS_DEBUG_CAPS_EXTENSIONS
#endif

#define SNIPPETS_EXTENSIONS                                  \
    OP_EXTENSION(ov::snippets::op::Brgemm)                   \
    OP_EXTENSION(ov::snippets::op::BroadcastLoad)            \
    OP_EXTENSION(ov::snippets::op::BroadcastMove)            \
    OP_EXTENSION(ov::snippets::op::ConvertSaturation)        \
    OP_EXTENSION(ov::snippets::op::ConvertTruncation)        \
    OP_EXTENSION(ov::snippets::op::Fill)                     \
    OP_EXTENSION(ov::snippets::op::HorizonMax)               \
    OP_EXTENSION(ov::snippets::op::HorizonSum)               \
    OP_EXTENSION(ov::snippets::op::KernelStatic)             \
    OP_EXTENSION(ov::snippets::op::KernelDynamic)            \
    OP_EXTENSION(ov::snippets::op::Load)                     \
    OP_EXTENSION(ov::snippets::op::LoadReshape)              \
    OP_EXTENSION(ov::snippets::op::LoopBegin)                \
    OP_EXTENSION(ov::snippets::op::LoopEnd)                  \
    OP_EXTENSION(ov::snippets::op::Buffer)                   \
    OP_EXTENSION(ov::snippets::op::Nop)                      \
    OP_EXTENSION(ov::snippets::op::PowerStatic)              \
    OP_EXTENSION(ov::snippets::op::Scalar)                   \
    OP_EXTENSION(ov::snippets::op::Store)                    \
    OP_EXTENSION(ov::snippets::op::Subgraph)                 \
    OP_EXTENSION(ov::snippets::op::VectorBuffer)             \
    OP_EXTENSION(ov::snippets::op::RankNormalization)        \
    OP_EXTENSION(ov::snippets::op::ReduceMax)                \
    OP_EXTENSION(ov::snippets::op::ReduceSum)                \
    OP_EXTENSION(ov::snippets::op::Reshape)

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>(
    {CPU_EXTENSIONS TYPE_RELAXED_EXTENSIONS SNIPPETS_EXTENSIONS SNIPPETS_DEBUG_CAPS_EXTENSIONS}));
