// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/ngraph.hpp>
#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <ov_ops/augru_cell.hpp>
#include <ov_ops/augru_sequence.hpp>
#include <ov_ops/multiclass_nms_ie_internal.hpp>
#include <ov_ops/nms_ie_internal.hpp>
#include <ov_ops/nms_static_shape_ie.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <snippets/op/subgraph.hpp>

#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "transformations/cpu_opset/common/op/leaky_relu.hpp"
#include "transformations/cpu_opset/common/op/ngram.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/cpu_opset/x64/op/interaction.hpp"
#include "transformations/cpu_opset/x64/op/mha.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    // cpu plugin opset
    std::make_shared<ov::OpExtension<ov::intel_cpu::FullyConnectedNode>>(),
        std::make_shared<ov::OpExtension<ov::intel_cpu::LeakyReluNode>>(),
        std::make_shared<ov::OpExtension<ov::intel_cpu::PowerStaticNode>>(),
        std::make_shared<ov::OpExtension<ov::intel_cpu::SwishNode>>(),
        std::make_shared<ov::OpExtension<ov::intel_cpu::NgramNode>>(),
#if defined(OPENVINO_ARCH_X86_64)
        std::make_shared<ov::OpExtension<ov::intel_cpu::MHANode>>(),
        std::make_shared<ov::OpExtension<ov::intel_cpu::InteractionNode>>(),
#endif
        // type relaxed opset
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::Add>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::AvgPool>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::Clamp>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::Concat>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::Convolution>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::ConvolutionBackpropData>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::DepthToSpace>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::Equal>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::FakeQuantize>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::Greater>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::GreaterEqual>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::GroupConvolution>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::GroupConvolutionBackpropData>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::Interpolate>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v4::Interpolate>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::Less>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::LessEqual>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::LogicalAnd>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::LogicalNot>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::LogicalOr>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::LogicalXor>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::MatMul>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::MaxPool>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::Multiply>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::NormalizeL2>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::NotEqual>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::PRelu>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::Relu>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::ReduceMax>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::ReduceLogicalAnd>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::ReduceLogicalOr>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::ReduceMean>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::ReduceMin>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::ReduceSum>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::Reshape>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::Select>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::ShapeOf>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::ShuffleChannels>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::Squeeze>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v1::Subtract>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::Unsqueeze>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v0::MVN>>>(),
        std::make_shared<ov::OpExtension<ov::op::TypeRelaxed<ngraph::op::v6::MVN>>>(),
        // internal opset
        std::make_shared<ov::OpExtension<ov::op::internal::NonMaxSuppressionIEInternal>>(),
        std::make_shared<ov::OpExtension<ov::op::internal::MulticlassNmsIEInternal>>(),
        std::make_shared<ov::OpExtension<ov::op::internal::AUGRUCell>>(),
        std::make_shared<ov::OpExtension<ov::op::internal::AUGRUSequence>>(),
        std::make_shared<ov::OpExtension<ov::op::internal::NmsStaticShapeIE<ov::op::v8::MatrixNms>>>(),
        // snippets opset
        std::make_shared<ov::OpExtension<ov::snippets::op::Brgemm>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::Buffer>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::BroadcastLoad>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::BroadcastMove>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::ConvertSaturation>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::ConvertTruncation>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::Fill>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::HorizonMax>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::HorizonSum>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::Kernel>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::Load>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::LoadReshape>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::LoopBegin>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::LoopEnd>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::Nop>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::PowerStatic>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::Scalar>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::Store>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::Subgraph>>(),
        std::make_shared<ov::OpExtension<ov::snippets::op::VectorBuffer>>(),
#if defined(OPENVINO_ARCH_X86_64)
        std::make_shared<ov::OpExtension<ov::intel_cpu::LoadConvertSaturation>>(),
        std::make_shared<ov::OpExtension<ov::intel_cpu::LoadConvertTruncation>>(),
        std::make_shared<ov::OpExtension<ov::intel_cpu::StoreConvertSaturation>>(),
        std::make_shared<ov::OpExtension<ov::intel_cpu::StoreConvertTruncation>>(),
        std::make_shared<ov::OpExtension<ov::intel_cpu::BrgemmCPU>>(),
        std::make_shared<ov::OpExtension<ov::intel_cpu::BrgemmCopyB>>(),
#endif
}));
