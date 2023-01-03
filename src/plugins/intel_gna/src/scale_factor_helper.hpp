// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "descriptions/gna_desc.hpp"
#include "serial/headers/latest/gna_model_header.hpp"

namespace GNAPluginNS {
class Config;
};  // namespace GNAPluginNS

namespace ov {
namespace intela_gna {
/**
 * @namespace helpers contains helpers tools for gna plugin.
 */
namespace helpers {

void ApplyInputScaleFactors(GNAPluginNS::GnaInputs& inputs,
                            const GNAPluginNS::Config& config,
                            const GNAPluginNS::HeaderLatest::ModelHeader& header);
void ApplyInputScaleFactors(GNAPluginNS::GnaInputs& inputs,
                            const GNAPluginNS::Config& config);

}  // namespace helpers
}  // namespace intela_gna
}  // namespace ov