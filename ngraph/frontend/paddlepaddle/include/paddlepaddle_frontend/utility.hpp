//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#pragma once

#include <frontend_manager/frontend_manager.hpp>

#include "ngraph/visibility.hpp"

#ifdef paddlepaddle_frontend_EXPORTS // defined if we are building the frontend_manager DLL (instead of using it)
#define PDPD_API NGRAPH_HELPER_DLL_EXPORT
#else
#define PDPD_API NGRAPH_HELPER_DLL_IMPORT
#endif // paddlepaddle_frontend_EXPORTS

namespace ngraph {
namespace frontend {

inline void PDPD_ASSERT(bool ex, const std::string& msg = "Unspecified error.") {
    if (!ex) throw std::runtime_error(msg);
}

#define PDPD_THROW(msg) throw std::runtime_error(std::string("ERROR: ") + msg)

#define NOT_IMPLEMENTED(msg) throw std::runtime_error(std::string(msg) + " is not implemented")

} // namespace frontend
} // namespace ngraph
