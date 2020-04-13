// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp"
#include <transformations/utils/pass_manager.hpp>
#include <memory>

bool ngraph::pass::ConvertOpSet1ToLegacy::run_on_function(std::shared_ptr<ngraph::Function> f) {
    auto pm = ngraph::pass::ConversionPassManager(transformation_callback);
    pm.run_passes(f);
    return true;
}