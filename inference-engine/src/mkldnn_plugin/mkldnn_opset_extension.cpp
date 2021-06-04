// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_opset_extension.h"
#include "ngraph_transformations/op/fully_connected.hpp"
#include "ngraph_transformations/op/leaky_relu.hpp"
#include "ngraph_transformations/op/power_static.hpp"
#include "ngraph_transformations/op/swish_cpu.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph_ops/type_relaxed.hpp>

#include <mutex>

namespace MKLDNNPlugin {

void OpsetExtension::GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept {
    static const InferenceEngine::Version version = {
        {1, 0},         // extension API version
        "1.0",
        "CPUPluginExt"  // extension description message
    };

    versionInfo = &version;
}

void OpsetExtension::Unload() noexcept {}

std::map<std::string, ngraph::OpSet> OpsetExtension::getOpSets() {
    std::map<std::string, ngraph::OpSet> opsets;

    ngraph::OpSet opset;

    opset.insert<FullyConnectedNode>();
    opset.insert<LeakyReluNode>();
    opset.insert<PowerStaticNode>();
    opset.insert<SwishNode>();

    opsets["cpu_plugin_opset"] = opset;

    return opsets;
}

std::vector<std::string> OpsetExtension::getImplTypes(const std::shared_ptr<ngraph::Node>&) {
    return {};
}

InferenceEngine::ILayerImpl::Ptr OpsetExtension::getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) {
    return nullptr;
}

void TypeRelaxedOpsetExtension::GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept {
    static const InferenceEngine::Version version = {
        {1, 0},                 // extension API version
        "1.0",
        "TypeRelaxedOpsetExt"   // extension description message
    };

    versionInfo = &version;
}

void TypeRelaxedOpsetExtension::Unload() noexcept {}

std::map<std::string, ngraph::OpSet> TypeRelaxedOpsetExtension::getOpSets() {
    static std::map<std::string, ngraph::OpSet> opsets;

    static std::once_flag flag;

    std::call_once(flag, [&]() {
        ngraph::OpSet opset;
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<ngraph::op::TypeRelaxed<NAMESPACE::NAME>>();
NGRAPH_OP(FullyConnectedNode, MKLDNNPlugin)
NGRAPH_OP(LeakyReluNode, MKLDNNPlugin)
NGRAPH_OP(PowerStaticNode, MKLDNNPlugin)
NGRAPH_OP(SwishNode, MKLDNNPlugin)
NGRAPH_OP(Abs, ngraph::op::v0)
NGRAPH_OP(Acos, ngraph::op::v0)
NGRAPH_OP(Add, ngraph::op::v1)
NGRAPH_OP(Asin, ngraph::op::v0)
NGRAPH_OP(Atan, ngraph::op::v0)
NGRAPH_OP(AvgPool, ngraph::op::v1)
NGRAPH_OP(BatchNormInference, ngraph::op::v0)
NGRAPH_OP(BinaryConvolution, ngraph::op::v1)
NGRAPH_OP(Broadcast, ngraph::op::v1)
NGRAPH_OP(CTCGreedyDecoder, ngraph::op::v0)
NGRAPH_OP(Ceiling, ngraph::op::v0)
NGRAPH_OP(Clamp, ngraph::op::v0)
NGRAPH_OP(Concat, ngraph::op::v0)
NGRAPH_OP(Constant, ngraph::op)
NGRAPH_OP(Convert, ngraph::op::v0)
NGRAPH_OP(ConvertLike, ngraph::op::v1)
NGRAPH_OP(Convolution, ngraph::op::v1)
NGRAPH_OP(ConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(Cos, ngraph::op::v0)
NGRAPH_OP(Cosh, ngraph::op::v0)
NGRAPH_OP(DeformableConvolution, ngraph::op::v1)
NGRAPH_OP(DeformablePSROIPooling, ngraph::op::v1)
NGRAPH_OP(DepthToSpace, ngraph::op::v0)
NGRAPH_OP(DetectionOutput, ngraph::op::v0)
NGRAPH_OP(Divide, ngraph::op::v1)
NGRAPH_OP(Elu, ngraph::op::v0)
NGRAPH_OP(Erf, ngraph::op::v0)
NGRAPH_OP(Equal, ngraph::op::v1)
NGRAPH_OP(Exp, ngraph::op::v0)
NGRAPH_OP(FakeQuantize, ngraph::op::v0)
NGRAPH_OP(Floor, ngraph::op::v0)
NGRAPH_OP(FloorMod, ngraph::op::v1)
NGRAPH_OP(Gather, ngraph::op::v1)
NGRAPH_OP(GatherTree, ngraph::op::v1)
NGRAPH_OP(Greater, ngraph::op::v1)
NGRAPH_OP(GreaterEqual, ngraph::op::v1)
NGRAPH_OP(GroupConvolution, ngraph::op::v1)
NGRAPH_OP(GroupConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(GRN, ngraph::op::v0)
NGRAPH_OP(HardSigmoid, ngraph::op::v0)
NGRAPH_OP(Interpolate, ngraph::op::v0)
NGRAPH_OP(Less, ngraph::op::v1)
NGRAPH_OP(LessEqual, ngraph::op::v1)
NGRAPH_OP(Log, ngraph::op::v0)
NGRAPH_OP(LogicalAnd, ngraph::op::v1)
NGRAPH_OP(LogicalNot, ngraph::op::v1)
NGRAPH_OP(LogicalOr, ngraph::op::v1)
NGRAPH_OP(LogicalXor, ngraph::op::v1)
NGRAPH_OP(LRN, ngraph::op::v0)
NGRAPH_OP(LSTMCell, ngraph::op::v0)
NGRAPH_OP(LSTMSequence, ngraph::op::v0)
NGRAPH_OP(MatMul, ngraph::op::v0)
NGRAPH_OP(MaxPool, ngraph::op::v1)
NGRAPH_OP(Maximum, ngraph::op::v1)
NGRAPH_OP(Minimum, ngraph::op::v1)
NGRAPH_OP(Mod, ngraph::op::v1)
NGRAPH_OP(Multiply, ngraph::op::v1)
NGRAPH_OP(Negative, ngraph::op::v0)
NGRAPH_OP(NonMaxSuppression, ngraph::op::v1)
NGRAPH_OP(NormalizeL2, ngraph::op::v0)
NGRAPH_OP(NotEqual, ngraph::op::v1)
NGRAPH_OP(OneHot, ngraph::op::v1)
NGRAPH_OP(PRelu, ngraph::op::v0)
NGRAPH_OP(PSROIPooling, ngraph::op::v0)
NGRAPH_OP(Pad, ngraph::op::v1)
NGRAPH_OP(Parameter, ngraph::op::v0)
NGRAPH_OP(Power, ngraph::op::v1)
NGRAPH_OP(PriorBox, ngraph::op::v0)
NGRAPH_OP(PriorBoxClustered, ngraph::op::v0)
NGRAPH_OP(Proposal, ngraph::op::v0)
NGRAPH_OP(Range, ngraph::op::v0)
NGRAPH_OP(Relu, ngraph::op::v0)
NGRAPH_OP(ReduceMax, ngraph::op::v1)
NGRAPH_OP(ReduceLogicalAnd, ngraph::op::v1)
NGRAPH_OP(ReduceLogicalOr, ngraph::op::v1)
NGRAPH_OP(ReduceMean, ngraph::op::v1)
NGRAPH_OP(ReduceMin, ngraph::op::v1)
NGRAPH_OP(ReduceProd, ngraph::op::v1)
NGRAPH_OP(ReduceSum, ngraph::op::v1)
NGRAPH_OP(RegionYolo, ngraph::op::v0)
NGRAPH_OP(Reshape, ngraph::op::v1)
NGRAPH_OP(Result, ngraph::op::v0)
NGRAPH_OP(Reverse, ngraph::op::v1)
NGRAPH_OP(ReverseSequence, ngraph::op::v0)
NGRAPH_OP(RNNCell, ngraph::op::v0)
NGRAPH_OP(Select, ngraph::op::v1)
NGRAPH_OP(Selu, ngraph::op::v0)
NGRAPH_OP(ShapeOf, ngraph::op::v0)
NGRAPH_OP(ShuffleChannels, ngraph::op::v0)
NGRAPH_OP(Sign, ngraph::op::v0)
NGRAPH_OP(Sigmoid, ngraph::op::v0)
NGRAPH_OP(Sin, ngraph::op::v0)
NGRAPH_OP(Sinh, ngraph::op::v0)
NGRAPH_OP(Softmax, ngraph::op::v1)
NGRAPH_OP(Sqrt, ngraph::op::v0)
NGRAPH_OP(SpaceToDepth, ngraph::op::v0)
NGRAPH_OP(Split, ngraph::op::v1)
NGRAPH_OP(SquaredDifference, ngraph::op::v0)
NGRAPH_OP(Squeeze, ngraph::op::v0)
NGRAPH_OP(StridedSlice, ngraph::op::v1)
NGRAPH_OP(Subtract, ngraph::op::v1)
NGRAPH_OP(Tan, ngraph::op::v0)
NGRAPH_OP(Tanh, ngraph::op::v0)
NGRAPH_OP(Tile, ngraph::op::v0)
NGRAPH_OP(TopK, ngraph::op::v1)
NGRAPH_OP(Transpose, ngraph::op::v1)
NGRAPH_OP(Unsqueeze, ngraph::op::v0)
NGRAPH_OP(VariadicSplit, ngraph::op::v1)
NGRAPH_OP(Xor, ngraph::op::v0)
#undef NGRAPH_OP
        opsets["type_relaxed_opset"] = opset;
    });

    return opsets;
}

std::vector<std::string> TypeRelaxedOpsetExtension::getImplTypes(const std::shared_ptr<ngraph::Node> &node) {
    return {};
}

InferenceEngine::ILayerImpl::Ptr TypeRelaxedOpsetExtension::getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) {
    return nullptr;
}

}  // namespace MKLDNNPlugin
