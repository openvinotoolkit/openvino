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

/**
 * @brief Defines openvino domains for tracing
 * @file itt.hpp
 */

#pragma once

#include <openvino/cc/factory.h>
#include <openvino/cc/selective_build.h>
#include <openvino/itt.hpp>

namespace ngraph
{
    namespace itt
    {
        namespace domains
        {
            OV_ITT_DOMAIN(nGraph);
            OV_ITT_DOMAIN(nGraphPass_LT);
            OV_ITT_DOMAIN(ngraph_op, "nGraph::Op");
        }
    }
}
OV_CC_DOMAINS(ngraph_op);
OV_ITT_DOMAIN(SIMPLE_ngraph_pass);

#if defined(SELECTIVE_BUILD_ANALYZER)
#define NGRAPH_OP_SCOPE(region) OV_SCOPE(ngraph_op, region)
#define NGRAPH_PASS_CALLBACK(matcher)                                                              \
    openvino::itt::handle_t m_callback_handle;                                                     \
    m_callback_handle = openvino::itt::handle(matcher->get_name());                                \
    OV_ITT_SCOPED_TASK(SIMPLE_ngraph_pass, m_callback_handle)
#elif defined(SELECTIVE_BUILD)
#define NGRAPH_OP_SCOPE(region)                                                                    \
    if (OV_CC_SCOPE_IS_ENABLED(OV_CC_CAT3(ngraph_op, _, region)) == 0)                             \
    throw ngraph::ngraph_error(std::string(OV_CC_TOSTRING(OV_CC_CAT3(ngraph_op, _, region))) +     \
                               " is disabled!")
#define NGRAPH_PASS_CALLBACK(matcher)
#else
#define NGRAPH_OP_SCOPE(region) OV_ITT_SCOPED_TASK(ngraph::itt::domains::ngraph_op, #region)
#define NGRAPH_PASS_CALLBACK(matcher)
#endif

#define NGRAPH_TYPE_CASE(region, a, ...)                                                           \
    case element::Type_t::a:                                                                       \
    {                                                                                              \
        OV_SCOPE(ngraph_op, OV_CC_CAT3(region, _, a))                                              \
        {                                                                                          \
            rc = evaluate<element::Type_t::a>(__VA_ARGS__);                                        \
        }                                                                                          \
    }                                                                                              \
    break

#define NGRAPH_COPY_TENSOR(region, a, ...)                                                         \
    case element::Type_t::a:                                                                       \
    {                                                                                              \
        OV_SCOPE(ngraph_op, OV_CC_CAT3(region, _, a))                                              \
        {                                                                                          \
            rc = copy_tensor<element::Type_t::a>(__VA_ARGS__);                                     \
        }                                                                                          \
    }                                                                                              \
    break
