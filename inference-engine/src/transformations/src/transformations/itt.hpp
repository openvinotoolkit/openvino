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

/**
 * @brief Defines openvino domains for tracing
 * @file itt.hpp
 */

#pragma once

#include <openvino/itt.hpp>

namespace ngraph {
namespace pass {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(IETransform);
}
}
}
}
#define OV_ITT_IE_TRANSFORM_CALLBACK(M, CN) OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::IETransform, "ngraph::pass::" + M.get_name() + "::" + CN)
#define OV_GEN_NGRAPH_PASS(NAME, FUNC) OV_ITT_GLUE_UNDERSCORE(Gen, OV_ITT_GLUE_UNDERSCORE(ngraph, OV_ITT_GLUE_UNDERSCORE(pass, OV_ITT_GLUE_UNDERSCORE(NAME, FUNC))))
