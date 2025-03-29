// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

// This transformation replaces pattern Parameter(Dict)->aten::__getitem__
class DictParameterResolver : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::pytorch::pass::DictParameterResolver");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};
// This transformation replaces pattern prim::DictConstruct->Result
class DictResultResolver : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::pytorch::pass::DictResultResolver");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
