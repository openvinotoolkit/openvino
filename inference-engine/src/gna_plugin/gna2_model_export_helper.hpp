// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if GNA_LIB_VER == 2

#include "gna2-model-suecreek-header.h"

#include <cstdint>
#include <string>

void * ExportSueLegacyUsingGnaApi2(
    uint32_t modelId,
    Gna2ModelSueCreekHeader* modelHeader);

void ExportLdForNoMmu(uint32_t modelId, std::ostream & outStream);
void ExportGnaDescriptorPartiallyFilled(uint32_t numberOfLayers, std::ostream & outStream);

#endif
