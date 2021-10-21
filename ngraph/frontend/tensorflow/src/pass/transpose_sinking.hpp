// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/pass/pass.hpp>

#include "tensorflow_frontend/utility.hpp"

namespace ov {
namespace frontend {
namespace tf {
namespace pass {

class TF_API TransposeSinkingOVTF : public ov::pass::FunctionPass {
public:
    OPENVINO_RTTI("ov::frontend::tf::pass::TransposeSinkingOVTF");
    TransposeSinkingOVTF() {
        set_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE, true);
    }
    bool run_on_function(std::shared_ptr<ov::Function> function) override;
};

}  // namespace pass
}  // namespace tf
}  // namespace frontend
}  // namespace ov
