//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/visibility.hpp"

#ifdef onnx_importer_EXPORTS // defined if we are building the ONNX_IMPORTER
#define ONNX_IMPORTER_API NGRAPH_HELPER_DLL_EXPORT
#else
#define ONNX_IMPORTER_API NGRAPH_HELPER_DLL_IMPORT
#endif
