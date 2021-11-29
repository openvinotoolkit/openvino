// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"

#ifdef onnx_importer_EXPORTS // defined if we are building the ONNX_IMPORTER
#define ONNX_IMPORTER_API NGRAPH_HELPER_DLL_EXPORT
#else
#define ONNX_IMPORTER_API NGRAPH_HELPER_DLL_IMPORT
#endif
