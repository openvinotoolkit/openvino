// Copyright (C) 2018 Intel Corporation
//
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
    void FuseConvolutionAndActivation(MKLDNNGraph &graph);
    void FuseConvolutionAndDepthwise(MKLDNNGraph &graph);
    void FuseConvolutionAndDWConvolution(MKLDNNGraph &graph);
    void FuseBatchNormWithScale(MKLDNNGraph& graph);
    void FuseConvolutionSumAndConvolutionSumActivation(MKLDNNGraph &graph);
    void RemoveIdentityOperator(MKLDNNGraph& graph);

    void RemoveIOScaleShifts(MKLDNNGraph& graph);
    void DropDoubleReorders(MKLDNNGraph& graph);

    void AddScaleShiftAfterInt8(MKLDNNGraph &graph);


    bool IsOneOf(Type type, std::vector<Type> types);
};

}  // namespace MKLDNNPlugin
