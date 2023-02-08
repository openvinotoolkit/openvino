// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pass.hpp>


namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

// This transformation handles GRUBlockCell with just one output - hidden state
class OPENVINO_API StructuralTypeProp : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::StructuralTypeProp");
    StructuralTypeProp();
};

// Replaces scalars of structural type Str by u8 dynamic tensors of rank 1
class OPENVINO_API ReplaceStrByU81D : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ReplaceStrByU81D");
    ReplaceStrByU81D();
};

// FIXME: Wrong name for transform. It lowers 1D dynamic tensors with strings to indexed u8 tensors
class OPENVINO_API DecomposeStrParameters : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::DecomposeStrParameters");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};


// Replaces scalars of structural type Str by u8 dynamic tensors of rank 1
class OPENVINO_API ThroughStrOpsProp : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ThroughStrOpsProp");
    ThroughStrOpsProp();
};

class OPENVINO_API ThroughReshapeProp : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ThroughReshapeProp");
    ThroughReshapeProp();
};

class OPENVINO_API ThroughNotEqualProp : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ThroughNotEqualProp");
    ThroughNotEqualProp();
};

class OPENVINO_API DecomposeStructResults : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::DecomposeStructResults");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

class OPENVINO_API ReplaceParameterByVocab : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ReplaceParameterByVocab");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

class OPENVINO_API ThroughWhileProp : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ThroughWhileProp");
    ThroughWhileProp();
};


class OPENVINO_API ThroughTensorListSetItem : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ThroughTensorListSetItem");
    ThroughTensorListSetItem();
};

class OPENVINO_API ThroughTensorListStack : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::ThroughTensorListStack");
    ThroughTensorListStack();
};




}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
