// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/moc_legacy_transformations.hpp"

#include <memory>

#include "ngraph/pass/manager.hpp"
#include "transformations/common_optimizations/change_placeholder_types.hpp"

bool ngraph::pass::MOCLegacyTransformations::run_on_function(
    std::shared_ptr<ov::Function> f) {
  ov::pass::Manager manager(get_pass_config());

  manager.register_pass<ngraph::pass::ChangePlaceholderTypes>();
  manager.run_passes(f);

  return false;
}
