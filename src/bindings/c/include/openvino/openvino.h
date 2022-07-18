// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file openvino.h
 * C API of OpenVINO 2.0 bridge unlocks using of OpenVINO 2.0
 * library and all its plugins in native applications disabling usage
 * of C++ API. The scope of API covers significant part of C++ API and includes
 * an ability to read model from the disk, modify input and output information
 * to correspond their runtime representation like data types or memory layout,
 * load in-memory model to different devices including
 * heterogeneous and multi-device modes, manage memory where input and output
 * is allocated and manage inference flow.
 **/
#pragma once

#include "ov_compiled_model.h"
#include "ov_core.h"
#include "ov_infer_request.h"
#include "ov_model.h"
#include "ov_prepostprocess.h"
#include "ov_property.h"
#include "ov_shape.h"
#include "ov_tensor.h"
