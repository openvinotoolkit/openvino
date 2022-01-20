// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/moc_legacy_transformations.hpp"

#include <memory>

#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/change_placeholder_types.hpp"

bool ov::pass::MOCLegacyTransformations::run_on_model(const std::shared_ptr<ov::Model>& f) {
    ov::pass::Manager manager(get_pass_config());

    manager.register_pass<ov::pass::ChangePlaceholderTypes>(m_params_with_custom_types);
    manager.run_passes(f);

    return false;
}
