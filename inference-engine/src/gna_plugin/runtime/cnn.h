// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <cstdio>

#include "backend/dnn_types.h"

#define CNN_MAX_POOL_SIZE 6

void CNNFilter32(intel_dnn_component_t *component);
void CNNMaxPool(intel_dnn_component_t *component, intel_dnn_number_type_t number_type, const bool sumPoolingOverRide = false);

#if GNA_LIB_VER == 2
void CNN2DFilter32(intel_dnn_component_t* component);
#endif
