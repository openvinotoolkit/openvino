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

namespace ngraph
{
    namespace itt
    {
        namespace domains
        {
            OV_ITT_DOMAIN(nGraph);
            OV_ITT_DOMAIN(nGraphOp, "nGraph::Op");
#if defined(OV_SELECTIVE_BUILD_LOG)
            OV_ITT_DOMAIN(CC_nGraphPassRegister);
            OV_ITT_DOMAIN(CC_nGraphPassAddMatcher);
            OV_ITT_DOMAIN(CC_nGraphPassCallback);
#endif
        }
    }
}

#if defined(OV_SELECTIVE_BUILD_LOG)
#define NGRAPH_DOMAIN OVConditionalCompilation::internal::itt::domains::CC0OV
#define NGRAPH_PASS_ADD_MATCHER_DOMAIN ngraph::itt::domains::CC_nGraphPassAddMatcher
#define NGRAPH_PASS_CALLBACK_DOMAIN ngraph::itt::domains::CC_nGraphPassCallback
#define NGRAPH_PASS_REGISTER_DOMAIN ngraph::itt::domains::CC_nGraphPassRegister
#else
#define NGRAPH_DOMAIN ngraph::itt::domains::nGraph
#define NGRAPH_PASS_ADD_MATCHER_DOMAIN ngraph::itt::domains::nGraph
#define NGRAPH_PASS_CALLBACK_DOMAIN ngraph::itt::domains::nGraph
#define NGRAPH_PASS_REGISTER_DOMAIN graph::itt::domains::nGraph
#endif


#if defined(OV_SELECTIVE_BUILD_LOG) || defined(ENABLE_PROFILING_ITT)
#define NGRAPH_TYPE_CASE(NAME, TYPE, ...)                                                          \
    case element::Type_t::TYPE: {                                                                  \
        OV_ITT_SCOPED_TASK(NGRAPH_DOMAIN, std::string(OV_TOSTRING(NAME ## _ ## TYPE)));            \
        rc = evaluate<element::Type_t::TYPE>(__VA_ARGS__);                                         \
        break;                                                                                     \
    }
#define NGRAPH_COPY_TENSOR(NAME, TYPE, ...)                                                        \
    case element::Type_t::TYPE: {                                                                  \
        OV_ITT_SCOPED_TASK(NGRAPH_DOMAIN, std::string(OV_TOSTRING(NAME ## _ ## TYPE)));            \
        rc = copy_tensor<element::Type_t::TYPE>(__VA_ARGS__);                                      \
        break;                                                                                     \
    }
#define NGRAPH_CASE(NAME, TYPE, ...)                                                               \
    case element::Type_t::TYPE: {                                                                  \
        const std::string case_name = std::string(OV_TOSTRING(NAME ## _ ## TYPE));                 \
        OV_ITT_SCOPED_TASK(NGRAPH_DOMAIN, case_name);                                              \
        __VA_ARGS__                                                                                \
    }
#else
#define NGRAPH_TYPE_CASE(NAME, TYPE, ...)                                                          \
    OV_SCOPE(OV_CAT(OV_CAT(NAME, _), TYPE),                                                        \
        TYPE_CASE(TYPE)(__VA_ARGS__);                                                              \
        break;                                                                                     \
    )
#define NGRAPH_COPY_TENSOR(NAME, TYPE, ...)                                                        \
    OV_SCOPE(OV_CAT(OV_CAT(NAME, _), TYPE),                                                        \
        COPY_TENSOR(TYPE)(__VA_ARGS__);                                                            \
        break;                                                                                     \
    )
#define NGRAPH_CASE(NAME, TYPE, ...)                                                               \
    OV_SCOPE(OV_CAT(OV_CAT(NAME, _), TYPE),                                                        \
        case element::Type_t::TYPE: {                                                              \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    )
#endif

#if defined(OV_SELECTIVE_BUILD_LOG) || defined(ENABLE_PROFILING_ITT)
    template<typename PASS>
    struct PassTag;
    #define NGRAPH_PASS_ADD_MATCHER(PASS)                                                                \
        OV_ITT_SCOPED_TASK(NGRAPH_PASS_ADD_MATCHER_DOMAIN,                                               \
            openvino::itt::handle<PassTag<T>>(this->get_class_name() +                                   \
                std::string("_") + PASS->get_name()))
    #define NGRAPH_PASS_CALLBACK(M) OV_ITT_SCOPED_TASK(NGRAPH_PASS_CALLBACK_DOMAIN, M->m_callback_handle)
    #define NGRAPH_PASS_SCOPE(region, ...)                                                              \
        OV_ITT_SCOPED_TASK(NGRAPH_PASS_REGISTER_DOMAIN, OV_TOSTRING(region));                           \
        __VA_ARGS__
#elif defined(OV_SELECTIVE_BUILD)
    #define NGRAPH_PASS_SCOPE(region, ...)                                                              \
        OV_EXPAND(OV_CAT(OV_SCOPE_, OV_SCOPE_IS_ENABLED(OV_CAT(REGISTER_PASS_, region)))(__VA_ARGS__))
    #define NGRAPH_PASS_ADD_MATCHER(...)
    #define NGRAPH_PASS_CALLBACK(...)
#else
    #define NGRAPH_PASS_ADD_MATCHER(...)
    #define NGRAPH_PASS_CALLBACK(...)
#endif

#define NGRAPH_OP_SCOPE(region, ...) OV_SCOPE(OV_CAT(ngraph_op_, region), __VA_ARGS__)
#define NGRAPH_OP_UTIL_SCOPE(region, ...) OV_SCOPE(OV_CAT(ngraph_op_util_, region), __VA_ARGS__)
