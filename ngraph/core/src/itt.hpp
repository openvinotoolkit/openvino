// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
        } // namespace domains
    }     // namespace itt
} // namespace ngraph
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
    if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ngraph_op, _, region)) == 0)                             \
    throw ngraph::ngraph_error(std::string(OV_PP_TOSTRING(OV_PP_CAT3(ngraph_op, _, region))) +     \
                               " is disabled!")
#define NGRAPH_PASS_CALLBACK(matcher)
#else
#define NGRAPH_OP_SCOPE(region)                                                                    \
    OV_ITT_SCOPED_TASK(ngraph::itt::domains::ngraph_op, OV_PP_TOSTRING(region))
#define NGRAPH_PASS_CALLBACK(matcher)
#endif

#define NGRAPH_TYPE_CASE(region, a, ...)                                                           \
    case element::Type_t::a:                                                                       \
    {                                                                                              \
        OV_SCOPE(ngraph_op, OV_PP_CAT3(region, _, a))                                              \
        {                                                                                          \
            rc = evaluate<element::Type_t::a>(__VA_ARGS__);                                        \
        }                                                                                          \
    }                                                                                              \
    break

#define NGRAPH_COPY_TENSOR(region, a, ...)                                                         \
    case element::Type_t::a:                                                                       \
    {                                                                                              \
        OV_SCOPE(ngraph_op, OV_PP_CAT3(region, _, a))                                              \
        {                                                                                          \
            rc = copy_tensor<element::Type_t::a>(__VA_ARGS__);                                     \
        }                                                                                          \
    }                                                                                              \
    break
