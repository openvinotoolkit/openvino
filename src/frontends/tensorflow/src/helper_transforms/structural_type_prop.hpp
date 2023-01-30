// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/tensorflow/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"


namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

// This transformation handles GRUBlockCell with just one output - hidden state
class StructuralTypeProp : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::StructuralTypeProp");
    StructuralTypeProp();
};

// Replaces scalars of structural type Str by u8 dynamic tensors of rank 1
class ReplaceStrByU81D : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ReplaceStrByU81D");
    ReplaceStrByU81D();
};

// FIXME: Wrong name for transform. It lowers 1D dynamic tensors with strings to indexed u8 tensors
class DecomposeStrParameters : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::DecomposeStrParameters");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};


// Replaces scalars of structural type Str by u8 dynamic tensors of rank 1
class ThroughStrOpsProp : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ThroughStrOpsProp");
    ThroughStrOpsProp();
};

class ThroughReshapeProp : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ThroughReshapeProp");
    ThroughReshapeProp();
};

class ThroughNotEqualProp : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ThroughNotEqualProp");
    ThroughNotEqualProp();
};

class DecomposeStructResults : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::DecomposeStructResults");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

class ReplaceParameterByVocab : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ReplaceParameterByVocab");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};


}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
