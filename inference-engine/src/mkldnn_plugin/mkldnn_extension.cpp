// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_extension.h"
#include "ngraph_transformations/op/fully_connected.hpp"
#include "ngraph_transformations/op/leaky_relu.hpp"
#include "ngraph_transformations/op/power_static.hpp"
#include "ngraph_transformations/op/swish_cpu.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph_ops/type_relaxed.hpp>

#include <mutex>

namespace MKLDNNPlugin {

void MKLDNNExtension::GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept {
    static const InferenceEngine::Version version = {
        {1, 0},             // extension API version
        "1.0",
        "MKLDNNExtension"   // extension description message
    };

    versionInfo = &version;
}

void MKLDNNExtension::Unload() noexcept {}

std::map<std::string, ngraph::OpSet> MKLDNNExtension::getOpSets() {
    auto cpu_plugin_opset = []() {
        ngraph::OpSet opset;

#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
        NGRAPH_OP(FullyConnectedNode, MKLDNNPlugin)
        NGRAPH_OP(LeakyReluNode, MKLDNNPlugin)
        NGRAPH_OP(PowerStaticNode, MKLDNNPlugin)
        NGRAPH_OP(SwishNode, MKLDNNPlugin)
#undef NGRAPH_OP

        return opset;
    };

    auto type_relaxed_opset = []() {
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

        return opset;
    };

    static std::map<std::string, ngraph::OpSet> opsets = {
        { "cpu_plugin_opset", cpu_plugin_opset() },
        { "type_relaxed_opset", type_relaxed_opset() }
    };

    return opsets;
}

std::vector<std::string> MKLDNNExtension::getImplTypes(const std::shared_ptr<ngraph::Node>&) {
    return {};
}

InferenceEngine::ILayerImpl::Ptr MKLDNNExtension::getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) {
    return nullptr;
}

}  // namespace MKLDNNPlugin

// Generate exported function
IE_DEFINE_EXTENSION_CREATE_FUNCTION(MKLDNNPlugin::MKLDNNExtension)

INFERENCE_EXTENSION_API(InferenceEngine::StatusCode)
InferenceEngine::CreateExtension(InferenceEngine::IExtension*& ext, InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        ext = new MKLDNNPlugin::MKLDNNExtension();
        return OK;
    } catch (std::exception& ex) {
        if (resp) {
            std::string err = ((std::string) "Couldn't create extension: ") + ex.what();
            err.copy(resp->msg, 255);
        }
        return InferenceEngine::GENERAL_ERROR;
    }
}
