// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/pass/pass.hpp>

#include "tensorflow_frontend/utility.hpp"

namespace ov {
namespace frontend {
namespace tf {
namespace pass {

/**
 * @brief Add OldApiMap with legacy type for Parameter node
 */
class TF_API ChangePlaceholderTypes : public ov::pass::FunctionPass {
public:
    OPENVINO_RTTI("ov::frontend::tf::pass::ChangePlaceholderTypes");
    ChangePlaceholderTypes() {}
    bool run_on_function(std::shared_ptr<ov::Function> function) override;
};

}  // namespace pass
}  // namespace tf
}  // namespace frontend
}  // namespace ov
