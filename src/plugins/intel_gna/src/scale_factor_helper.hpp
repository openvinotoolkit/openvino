// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "descriptions/gna_desc.hpp"
#include "serial/headers/latest/gna_model_header.hpp"

namespace ov {
namespace intel_gna {

class Config;

/**
 * @namespace helpers contains helpers tools for gna plugin.
 */
namespace helpers {

void ApplyInputScaleFactors(const Config& config,
                            const header_latest::ModelHeader& header,
                            GnaInputs& inputs);

}  // namespace helpers
}  // namespace intel_gna
}  // namespace ov