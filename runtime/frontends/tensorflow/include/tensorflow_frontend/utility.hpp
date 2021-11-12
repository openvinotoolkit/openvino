// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "frontend_manager/frontend_exceptions.hpp"

#ifdef ov_tensorflow_frontend_EXPORTS
#    define TF_API OPENVINO_CORE_EXPORTS
#else
#    define TF_API OPENVINO_CORE_IMPORTS
#endif  // ov_tensorflow_frontend_EXPORTS
