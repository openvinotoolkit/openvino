// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension.h"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "transformations/cpu_opset/common/op/leaky_relu.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/cpu_opset/common/op/ngram.hpp"
#include "transformations/cpu_opset/x64/op/mha.hpp"
#include "transformations/cpu_opset/x64/op/interaction.hpp"
#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"

#include <ngraph/ngraph.hpp>
#include <ov_ops/augru_cell.hpp>
#include <ov_ops/augru_sequence.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <ov_ops/nms_ie_internal.hpp>
#include <ov_ops/nms_static_shape_ie.hpp>
#include <ov_ops/multiclass_nms_ie_internal.hpp>

#include <snippets/op/subgraph.hpp>

#include <mutex>

namespace ov {
namespace intel_cpu {

void Extension::GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept {
    static const InferenceEngine::Version version = {
        {1, 0},             // extension API version
        "1.0",
        "Extension"   // extension description message
    };

    versionInfo = &version;
}

void Extension::Unload() noexcept {}

std::map<std::string, ngraph::OpSet> Extension::getOpSets() {
    auto cpu_plugin_opset = []() {
        ngraph::OpSet opset;

#if defined(OPENVINO_ARCH_X86_64)
#define NGRAPH_OP_X64(NAME, NAMESPACE) NGRAPH_OP(NAME, NAMESPACE)
#else
#define NGRAPH_OP_X64(NAME, NAMESPACE)
#endif

#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
        NGRAPH_OP(FullyConnectedNode, ov::intel_cpu)
        NGRAPH_OP(LeakyReluNode, ov::intel_cpu)
        NGRAPH_OP(PowerStaticNode, ov::intel_cpu)
        NGRAPH_OP(SwishNode, ov::intel_cpu)
        NGRAPH_OP(NgramNode, ov::intel_cpu)
        NGRAPH_OP_X64(MHANode, ov::intel_cpu)
        NGRAPH_OP_X64(InteractionNode, ov::intel_cpu)
#undef NGRAPH_OP

        return opset;
    };

    auto type_relaxed_opset = []() {
        ngraph::OpSet opset;

#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<ov::op::TypeRelaxed<NAMESPACE::NAME>>();
        NGRAPH_OP(Add, ngraph::op::v1)
        NGRAPH_OP(AvgPool, ngraph::op::v1)
        NGRAPH_OP(Clamp, ngraph::op::v0)
        NGRAPH_OP(Concat, ngraph::op::v0)
        NGRAPH_OP(Convolution, ngraph::op::v1)
        NGRAPH_OP(ConvolutionBackpropData, ngraph::op::v1)
        NGRAPH_OP(DepthToSpace, ngraph::op::v0)
        NGRAPH_OP(Equal, ngraph::op::v1)
        NGRAPH_OP(FakeQuantize, ngraph::op::v0)
        NGRAPH_OP(Greater, ngraph::op::v1)
        NGRAPH_OP(GreaterEqual, ngraph::op::v1)
        NGRAPH_OP(GroupConvolution, ngraph::op::v1)
        NGRAPH_OP(GroupConvolutionBackpropData, ngraph::op::v1)
        NGRAPH_OP(Interpolate, ngraph::op::v0)
        NGRAPH_OP(Interpolate, ngraph::op::v4)
        NGRAPH_OP(Less, ngraph::op::v1)
        NGRAPH_OP(LessEqual, ngraph::op::v1)
        NGRAPH_OP(LogicalAnd, ngraph::op::v1)
        NGRAPH_OP(LogicalNot, ngraph::op::v1)
        NGRAPH_OP(LogicalOr, ngraph::op::v1)
        NGRAPH_OP(LogicalXor, ngraph::op::v1)
        NGRAPH_OP(MatMul, ngraph::op::v0)
        NGRAPH_OP(MaxPool, ngraph::op::v1)
        NGRAPH_OP(Multiply, ngraph::op::v1)
        NGRAPH_OP(NormalizeL2, ngraph::op::v0)
        NGRAPH_OP(NotEqual, ngraph::op::v1)
        NGRAPH_OP(PRelu, ngraph::op::v0)
        NGRAPH_OP(Relu, ngraph::op::v0)
        NGRAPH_OP(ReduceMax, ngraph::op::v1)
        NGRAPH_OP(ReduceLogicalAnd, ngraph::op::v1)
        NGRAPH_OP(ReduceLogicalOr, ngraph::op::v1)
        NGRAPH_OP(ReduceMean, ngraph::op::v1)
        NGRAPH_OP(ReduceMin, ngraph::op::v1)
        NGRAPH_OP(ReduceSum, ngraph::op::v1)
        NGRAPH_OP(Reshape, ngraph::op::v1)
        NGRAPH_OP(Select, ngraph::op::v1)
        NGRAPH_OP(ShapeOf, ngraph::op::v0)
        NGRAPH_OP(ShuffleChannels, ngraph::op::v0)
        NGRAPH_OP(Squeeze, ngraph::op::v0)
        NGRAPH_OP(Subtract, ngraph::op::v1)
        NGRAPH_OP(Unsqueeze, ngraph::op::v0)
        NGRAPH_OP(MVN, ngraph::op::v0)
        NGRAPH_OP(MVN, ngraph::op::v6)
        NGRAPH_OP(Select, ngraph::op::v1)
        NGRAPH_OP(ConvolutionBackpropData, ngraph::op::v1)
#undef NGRAPH_OP

        return opset;
    };

    auto ie_internal_opset = []() {
        ngraph::OpSet opset;

#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
        NGRAPH_OP(NonMaxSuppressionIEInternal, ov::op::internal)
        NGRAPH_OP(MulticlassNmsIEInternal, ov::op::internal)
        NGRAPH_OP(AUGRUCell, ov::op::internal)
        NGRAPH_OP(AUGRUSequence, ov::op::internal)
        NGRAPH_OP(NmsStaticShapeIE<ov::op::v8::MatrixNms>, ov::op::internal)
#undef NGRAPH_OP

        return opset;
    };

    auto snippets_opset = []() {
        ngraph::OpSet opset;

#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
        NGRAPH_OP(Brgemm, ov::snippets::op)
        NGRAPH_OP(Buffer, ov::snippets::op)
        NGRAPH_OP(BroadcastLoad, ov::snippets::op)
        NGRAPH_OP(BroadcastMove, ov::snippets::op)
        NGRAPH_OP(ConvertSaturation, ov::snippets::op)
        NGRAPH_OP(ConvertTruncation, ov::snippets::op)
        NGRAPH_OP(Fill, ov::snippets::op)
        NGRAPH_OP(HorizonMax, ov::snippets::op)
        NGRAPH_OP(HorizonSum, ov::snippets::op)
        NGRAPH_OP(Kernel, ov::snippets::op)
        NGRAPH_OP(Load, ov::snippets::op)
        NGRAPH_OP(LoadReshape, ov::snippets::op)
        NGRAPH_OP(LoopBegin, ov::snippets::op)
        NGRAPH_OP(LoopEnd, ov::snippets::op)
        NGRAPH_OP(Nop, ov::snippets::op)
        NGRAPH_OP(PowerStatic, ov::snippets::op)
        NGRAPH_OP(Scalar, ov::snippets::op)
        NGRAPH_OP(Store, ov::snippets::op)
        NGRAPH_OP(Subgraph, ov::snippets::op)
        NGRAPH_OP(VectorBuffer, ov::snippets::op)
        NGRAPH_OP(RankNormalization, ov::snippets::op)
        NGRAPH_OP_X64(LoadConvertSaturation, ov::intel_cpu)
        NGRAPH_OP_X64(LoadConvertTruncation, ov::intel_cpu)
        NGRAPH_OP_X64(StoreConvertSaturation, ov::intel_cpu)
        NGRAPH_OP_X64(StoreConvertTruncation, ov::intel_cpu)
        NGRAPH_OP_X64(BrgemmCPU, ov::intel_cpu)
        NGRAPH_OP_X64(BrgemmCopyB, ov::intel_cpu)
#undef NGRAPH_OP

        return opset;
    };

    static std::map<std::string, ngraph::OpSet> opsets = {
        { "cpu_plugin_opset", cpu_plugin_opset() },
        { "type_relaxed_opset", type_relaxed_opset() },
        { "ie_internal_opset", ie_internal_opset() },
        { "SnippetsOpset", snippets_opset() },
    };

    return opsets;
}

std::vector<std::string> Extension::getImplTypes(const std::shared_ptr<ngraph::Node>&) {
    return {};
}

InferenceEngine::ILayerImpl::Ptr Extension::getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) {
    return nullptr;
}

}   // namespace intel_cpu
}   // namespace ov

// Generate exported function
IE_DEFINE_EXTENSION_CREATE_FUNCTION(ov::intel_cpu::Extension)
