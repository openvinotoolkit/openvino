// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

class GnaFuseMarkUpNodesOrder : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GnaFuseMarkUpNodesOrder", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class GnaFuseCleanUpNodesOrder : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GnaFuseCleanUpNodesOrder", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class FuseConvolutionWithBiasAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseConvolutionWithBiasAdd", "0");
    FuseConvolutionWithBiasAdd();
};

class FuseGroupConvolutionWithBiasAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseGroupConvolutionWithBiasAdd", "0");
    FuseGroupConvolutionWithBiasAdd();
};

class FuseConvolutionWithBiasAddAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseConvolutionWithBiasAddAdd", "0");
    FuseConvolutionWithBiasAddAdd();
};

class SinkActivationToGnaConvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SinkActivationToGnaConvolution", "0");
    SinkActivationToGnaConvolution();
};

class GnaConvolutionFusion : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GnaConvolutionFusion", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
