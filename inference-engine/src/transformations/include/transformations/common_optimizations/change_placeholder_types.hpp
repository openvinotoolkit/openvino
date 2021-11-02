// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ChangePlaceholderTypes;

/**
 * @brief Add OldApiMap with legacy type for Parameter node
 */
class ChangePlaceholderTypes : public ngraph::pass::FunctionPass {
 public:
  NGRAPH_RTTI_DECLARATION;
  bool run_on_function(std::shared_ptr<ov::Function> function) override;
};

}  // namespace pass
}  // namespace ngraph
