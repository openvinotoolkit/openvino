// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API NopElimination;

}  // namespace pass
}  // namespace ov

// TODO: add description here
class ov::pass::NopElimination : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("NopElimination", "0");
    explicit NopElimination(bool use_shape_for_elimination = true) :
            ov::pass::ModelPass(),
            _use_shape_for_elimination(use_shape_for_elimination) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
private:
    const bool _use_shape_for_elimination;
};
