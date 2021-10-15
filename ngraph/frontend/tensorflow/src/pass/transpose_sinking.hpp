// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/pass/pass.hpp>

namespace ov {
namespace frontend {
namespace tf {
namespace pass {

class TransposeSinkingOVTF : public ov::pass::FunctionPass {
public:
    TransposeSinkingOVTF() {
        set_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE, true);
    }
    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};

}  // namespace pass
}  // namespace tf
}  // namespace frontend
}  // namespace ov
