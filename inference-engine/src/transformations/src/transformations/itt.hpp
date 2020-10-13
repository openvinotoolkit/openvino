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
#include <openvino/conditional_compilation.hpp>
#if defined(OV_SELECTIVE_BUILD)
#include "ov_gen.hpp"
#endif

namespace ngraph {
namespace pass {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(IETransform);
#if defined(OV_SELECTIVE_BUILD_LOG)
    OV_ITT_DOMAIN(CC_nGraphPassRegister);
#endif
}
}
}
}

#if defined(OV_SELECTIVE_BUILD)
#define IETRANSFORM_SCOPE(region, ...)                                                             \
    std::string matcher_name(OV_TOSTRING(region));                                                 \
    OV_EXPAND(OV_CAT(OV_SCOPE_, OV_SCOPE_IS_ENABLED(OV_CAT(REGISTER_PASS_, region)))(__VA_ARGS__))
#elif defined(OV_SELECTIVE_BUILD_LOG)
#define IETRANSFORM_SCOPE(region, ...)                                                             \
    std::string matcher_name(OV_TOSTRING(region));                                                 \
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::CC_nGraphPassRegister,                          \
        openvino::itt::handle(matcher_name));                                                      \
    __VA_ARGS__
#else
#define IETRANSFORM_SCOPE(region, ...)                                                             \
    std::string matcher_name(OV_TOSTRING(region));                                                 \
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::IETransform,                                    \
        openvino::itt::handle("ngraph::pass::" + matcher_name));                                   \
    __VA_ARGS__
#endif