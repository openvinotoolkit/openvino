// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_graph.h"
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNGraphOptimizer {
public:
    MKLDNNGraphOptimizer();

public:
    void ApplyCommonGraphOptimizations(MKLDNNGraph& graph);
    void ApplyImplSpecificGraphOptimizations(MKLDNNGraph& graph);

private:
    void SLTMTransform(MKLDNNGraph& graph);
    void MergeGroupConvolution(MKLDNNGraph& graph);
#if defined(COMPILED_CPU_MKLDNN_ACTIVATION_NODE)
    void FuseConvolutionAndActivation(MKLDNNGraph &graph);
    void FuseFullyConnectedAndActivation(MKLDNNGraph &graph);
#endif
#if defined (COMPILED_CPU_MKLDNN_DEPTHWISE_NODE)
    void FuseConvolutionAndDepthwise(MKLDNNGraph &graph);
#endif
    void FuseConvolutionAndDWConvolution(MKLDNNGraph &graph);
#if defined(COMPILED_CPU_MKLDNN_QUANTIZE_NODE)
    void FuseBinaryConvolutionAndQuantize(MKLDNNGraph &graph);
#endif
    void FuseBatchNormWithScale(MKLDNNGraph& graph);
#if defined(COMPILED_CPU_MKLDNN_ELTWISE_NODE)
    void FuseConvolutionSumAndConvolutionSumActivation(MKLDNNGraph &graph);
#endif
    void RemoveIdentityOperator(MKLDNNGraph& graph);

    void RemoveIOScaleShifts(MKLDNNGraph& graph);
#if defined (COMPILED_CPU_MKLDNN_REORDER_NODE)
    void DropDoubleReorders(MKLDNNGraph& graph);
#endif

    void AddScaleShiftAfterInt8(MKLDNNGraph &graph);


    bool IsOneOf(Type type, std::vector<Type> types);
};

}  // namespace MKLDNNPlugin
