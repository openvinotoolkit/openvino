// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna_data_types.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

class MarkOutputSinking : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MarkOutputSinking", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
