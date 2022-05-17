/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once


#define DNNL_MACRO_EXPAND(x) x

#define DNNL_MACRO_CAT_(x, y) x ## y
#define DNNL_MACRO_CAT(x, y) DNNL_MACRO_CAT_(x, y)
#define DNNL_MACRO_CAT3_(x, y, z) x ## y ## z
#define DNNL_MACRO_CAT3(x, y, z) DNNL_MACRO_CAT3_(x, y, z)

#define DNNL_MACRO_TOSTRING(...) DNNL_MACRO_TOSTRING_(__VA_ARGS__)
#define DNNL_MACRO_TOSTRING_(...) #__VA_ARGS__

#define DNNL_MACRO_NARG(...) DNNL_MACRO_EXPAND( DNNL_MACRO_NARG_(__VA_ARGS__, DNNL_MACRO_RSEQ_N()) )
#define DNNL_MACRO_NARG_(...) DNNL_MACRO_EXPAND( DNNL_MACRO_ARG_N(__VA_ARGS__) )
#define DNNL_MACRO_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define DNNL_MACRO_RSEQ_N() 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define DNNL_MACRO_EVAL_(NAME, N) NAME ## _ ## N
#define DNNL_MACRO_EVAL(NAME, N) DNNL_MACRO_EVAL_(NAME, N)

#define DNNL_MACRO_OVERLOAD(NAME, ...) \
    DNNL_MACRO_EXPAND( DNNL_MACRO_EVAL(NAME, DNNL_MACRO_EXPAND( DNNL_MACRO_NARG(__VA_ARGS__) ))(__VA_ARGS__) )

#if defined(SELECTIVE_BUILD_ANALYZER)

# include <openvino/cc/selective_build.h>

namespace dnnl {

OV_CC_DOMAINS(DNNL)

}   // namespace dnnl

# define DNNL_CSCOPE(region) OV_SCOPE(DNNL, region)

# define DNNL_PRIMITIVE_NAME_INIT(pd_t) name = typeid(pd_t).name();
# define DNNL_PRIMITIVE_CREATE(pd_t) OV_ITT_SCOPED_TASK(dnnl::FACTORY_DNNL, std::string("CREATE$CPUEngine$") + typeid(pd_t).name());
# define DNNL_PRIMITIVE_IMPL(...) DNNL_MACRO_OVERLOAD(DNNL_PRIMITIVE_IMPL, __VA_ARGS__),
# define DNNL_PRIMITIVE_IMPL_2(expr, type) dnnl::impl::move(expr(type), OV_CC_TOSTRING(type))
# define DNNL_PRIMITIVE_IMPL_3(expr, type, t1) dnnl::impl::move(expr(type<t1>), OV_CC_TOSTRING(type ## _ ## t1))
# define DNNL_PRIMITIVE_IMPL_4(expr, type, t1, t2) dnnl::impl::move(expr(type<t1, t2>), OV_CC_TOSTRING(type ## _ ## t1 ## _ ## t2))
# define DNNL_PRIMITIVE_IMPL_5(expr, type, t1, t2, t3) dnnl::impl::move(expr(type<t1, t2, t3>), OV_CC_TOSTRING(type ## _ ## t1 ## _ ## t2 ## _ ## t3))
# define DNNL_PRIMITIVE_IMPL_6(expr, type, t1, t2, t3, t4) dnnl::impl::move(expr(type<t1, t2, t3, t4>), OV_CC_TOSTRING(type ## _ ## t1 ## _ ## t2 ## _ ## t3 ## _ ## t4))
# define DNNL_PRIMITIVE_IMPL_7(expr, type, t1, t2, t3, t4, t5) dnnl::impl::move(expr(type<t1, t2, t3, t4, t5>), OV_CC_TOSTRING(type ## _ ## t1 ## _ ## t2 ## _ ## t3 ## _ ## t4 ## _ ## t5))
# define DNNL_PRIMITIVE_IMPL_8(expr, type, t1, t2, t3, t4, t5, t6) dnnl::impl::move(expr(type<t1, t2, t3, t4, t5, t6>), OV_CC_TOSTRING(type ## _ ## t1 ## _ ## t2 ## _ ## t3 ## _ ## t4 ## _ ## t5 ## _ ## t6))
# define DNNL_PRIMITIVE_IMPL_9(expr, type, t1, t2, t3, t4, t5, t6, t7) dnnl::impl::move(expr(type<t1, t2, t3, t4, t5, t6, t7>), OV_CC_TOSTRING(type ## _ ## t1 ## _ ## t2 ## _ ## t3 ## _ ## t4 ## _ ## t5 ## _ ## t6 ## _ ## t7))

#elif defined(SELECTIVE_BUILD)

# include <openvino/cc/selective_build.h>

# define DNNL_CSCOPE(region) OV_SCOPE(DNNL, region)

# define DNNL_OBJ_BUILDER_0(...)
# define DNNL_OBJ_BUILDER_1(...) __VA_ARGS__,
# define DNNL_OBJ_BUILDER(name, ...) OV_CC_EXPAND(OV_CC_CAT(DNNL_OBJ_BUILDER_, OV_CC_EXPAND(OV_CC_SCOPE_IS_ENABLED(OV_CC_CAT(DNNL_, name))))(__VA_ARGS__))

# define DNNL_PRIMITIVE_NAME_INIT(pd_t)
# define DNNL_PRIMITIVE_CREATE(pd_t)
# define DNNL_PRIMITIVE_IMPL(...) DNNL_MACRO_OVERLOAD(DNNL_PRIMITIVE_IMPL, __VA_ARGS__)
# define DNNL_PRIMITIVE_IMPL_2(expr, type) DNNL_OBJ_BUILDER(type, expr(type))
# define DNNL_PRIMITIVE_IMPL_3(expr, type, t1) DNNL_OBJ_BUILDER(type ## _ ## t1, expr(type<t1>))
# define DNNL_PRIMITIVE_IMPL_4(expr, type, t1, t2) DNNL_OBJ_BUILDER(type ## _ ## t1 ## _ ## t2, expr(type<t1, t2>))
# define DNNL_PRIMITIVE_IMPL_5(expr, type, t1, t2, t3) DNNL_OBJ_BUILDER(type ## _ ## t1 ## _ ## t2 ## _ ## t3, expr(type<t1, t2, t3>))
# define DNNL_PRIMITIVE_IMPL_6(expr, type, t1, t2, t3, t4) DNNL_OBJ_BUILDER(type ## _ ## t1 ## _ ## t2 ## _ ## t3 ## _ ## t4, expr(type<t1, t2, t3, t4>))
# define DNNL_PRIMITIVE_IMPL_7(expr, type, t1, t2, t3, t4, t5) DNNL_OBJ_BUILDER(type ## _ ## t1 ## _ ## t2 ## _ ## t3 ## _ ## t4 ## _ ## t5, expr(type<t1, t2, t3, t4, t5>))
# define DNNL_PRIMITIVE_IMPL_8(expr, type, t1, t2, t3, t4, t5, t6) DNNL_OBJ_BUILDER(type ## _ ## t1 ## _ ## t2 ## _ ## t3 ## _ ## t4 ## _ ## t5 ## _ ## t6, expr(type<t1, t2, t3, t4, t5, t6>))
# define DNNL_PRIMITIVE_IMPL_9(expr, type, t1, t2, t3, t4, t5, t6, t7) DNNL_OBJ_BUILDER(type ## _ ## t1 ## _ ## t2 ## _ ## t3 ## _ ## t4 ## _ ## t5 ## _ ## t6 ## _ ## t7, expr(type<t1, t2, t3, t4, t5, t6, t7>))

#else

# define DNNL_CSCOPE(region)

# define DNNL_PRIMITIVE_NAME_INIT(pd_t)
# define DNNL_PRIMITIVE_CREATE(pd_t)
# define DNNL_PRIMITIVE_IMPL(...) DNNL_MACRO_OVERLOAD(DNNL_PRIMITIVE_IMPL, __VA_ARGS__),
# define DNNL_PRIMITIVE_IMPL_2(expr, type) expr(type)
# define DNNL_PRIMITIVE_IMPL_3(expr, type, t1) expr(type<t1>)
# define DNNL_PRIMITIVE_IMPL_4(expr, type, t1, t2) expr(type<t1, t2>)
# define DNNL_PRIMITIVE_IMPL_5(expr, type, t1, t2, t3) expr(type<t1, t2, t3>)
# define DNNL_PRIMITIVE_IMPL_6(expr, type, t1, t2, t3, t4) expr(type<t1, t2, t3, t4>)
# define DNNL_PRIMITIVE_IMPL_7(expr, type, t1, t2, t3, t4, t5) expr(type<t1, t2, t3, t4, t5>)
# define DNNL_PRIMITIVE_IMPL_8(expr, type, t1, t2, t3, t4, t5, t6) expr(type<t1, t2, t3, t4, t5, t6>)
# define DNNL_PRIMITIVE_IMPL_9(expr, type, t1, t2, t3, t4, t5, t6, t7) expr(type<t1, t2, t3, t4, t5, t6, t7>)

#endif
