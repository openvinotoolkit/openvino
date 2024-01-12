// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include "openvino/core/op_extension.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "snippets/op/subgraph.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "transformations/cpu_opset/common/op/leaky_relu.hpp"
#include "transformations/cpu_opset/common/op/ngram.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/cpu_opset/x64/op/interaction.hpp"
#include "transformations/cpu_opset/x64/op/mha.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/perf_count_rdtsc.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"

#define OP_EXTENSION(NAME) std::make_shared<ov::OpExtension<NAME>>(),

#if defined(OPENVINO_ARCH_X86_64)
#    define OP_EXTENSION_X64(NAME) OP_EXTENSION(NAME)
#else
#    define OP_EXTENSION_X64(NAME)
#endif

#define CPU_EXTENSIONS                                                      \
    OP_EXTENSION(ov::intel_cpu::FullyConnectedNode)                         \
    OP_EXTENSION(ov::intel_cpu::LeakyReluNode)                              \
    OP_EXTENSION(ov::intel_cpu::PowerStaticNode)                            \
    OP_EXTENSION(ov::intel_cpu::SwishNode)                                  \
    OP_EXTENSION(ov::intel_cpu::NgramNode)                                  \
    OP_EXTENSION(ov::op::internal::NonMaxSuppressionIEInternal)             \
    OP_EXTENSION(ov::op::internal::MulticlassNmsIEInternal)                 \
    OP_EXTENSION(ov::op::internal::AUGRUCell)                               \
    OP_EXTENSION(ov::op::internal::AUGRUSequence)                           \
    OP_EXTENSION(ov::op::internal::NmsStaticShapeIE<ov::op::v8::MatrixNms>) \
    OP_EXTENSION_X64(ov::intel_cpu::MHANode)                                \
    OP_EXTENSION_X64(ov::intel_cpu::InteractionNode)                        \
    OP_EXTENSION_X64(ov::intel_cpu::ScaledDotProductAttentionWithKVCache)   \
    OP_EXTENSION_X64(ov::intel_cpu::LoadConvertSaturation)                  \
    OP_EXTENSION_X64(ov::intel_cpu::LoadConvertTruncation)                  \
    OP_EXTENSION_X64(ov::intel_cpu::StoreConvertSaturation)                 \
    OP_EXTENSION_X64(ov::intel_cpu::StoreConvertTruncation)                 \
    OP_EXTENSION_X64(ov::intel_cpu::BrgemmCPU)                              \
    OP_EXTENSION_X64(ov::intel_cpu::BrgemmCopyB)

#define TYPE_RELAXED_EXTENSIONS                                                 \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::Add>)                          \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::AvgPool>)                      \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::Clamp>)                        \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::Concat>)                       \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::Convolution>)                  \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::ConvolutionBackpropData>)      \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::DepthToSpace>)                 \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::Equal>)                        \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::FakeQuantize>)                 \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::Greater>)                      \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::GreaterEqual>)                 \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::GroupConvolution>)             \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::GroupConvolutionBackpropData>) \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::Interpolate>)                  \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v4::Interpolate>)                  \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::Less>)                         \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::LessEqual>)                    \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::LogicalAnd>)                   \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::LogicalNot>)                   \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::LogicalOr>)                    \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::LogicalXor>)                   \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::MatMul>)                       \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::MaxPool>)                      \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::Multiply>)                     \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::NormalizeL2>)                  \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::NotEqual>)                     \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::PRelu>)                        \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::Relu>)                         \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::ReduceMax>)                    \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::ReduceLogicalAnd>)             \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::ReduceLogicalOr>)              \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::ReduceMean>)                   \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::ReduceMin>)                    \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::ReduceSum>)                    \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::Reshape>)                      \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::Select>)                       \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::ShapeOf>)                      \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::ShuffleChannels>)              \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::Squeeze>)                      \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v1::Subtract>)                     \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::Unsqueeze>)                    \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v0::MVN>)                          \
    OP_EXTENSION(ov::op::TypeRelaxed<ov::op::v6::MVN>)

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
    OP_EXTENSION(ov::snippets::op::Kernel)                   \
    OP_EXTENSION(ov::snippets::op::IntermediateMemoryBuffer) \
    OP_EXTENSION(ov::snippets::op::Load)                     \
    OP_EXTENSION(ov::snippets::op::LoadReshape)              \
    OP_EXTENSION(ov::snippets::op::LoopBegin)                \
    OP_EXTENSION(ov::snippets::op::LoopEnd)                  \
    OP_EXTENSION(ov::snippets::op::NewMemoryBuffer)          \
    OP_EXTENSION(ov::snippets::op::Nop)                      \
    OP_EXTENSION(ov::snippets::op::PowerStatic)              \
    OP_EXTENSION(ov::snippets::op::Scalar)                   \
    OP_EXTENSION(ov::snippets::op::Store)                    \
    OP_EXTENSION(ov::snippets::op::Subgraph)                 \
    OP_EXTENSION(ov::snippets::op::VectorBuffer)             \
    OP_EXTENSION(ov::snippets::op::RankNormalization)

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>(
    {CPU_EXTENSIONS TYPE_RELAXED_EXTENSIONS SNIPPETS_EXTENSIONS SNIPPETS_DEBUG_CAPS_EXTENSIONS}));
