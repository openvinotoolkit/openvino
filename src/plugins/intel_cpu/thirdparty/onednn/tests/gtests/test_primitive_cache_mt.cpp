/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "dnnl.hpp"

namespace dnnl {

TEST(primitive_cache_mt_test, TestGeneralCase) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    engine eng(get_test_engine_kind(), 0);

    // Flush the cache
    set_primitive_cache_capacity(0);
    set_primitive_cache_capacity(1024);

    memory::dim n_primitives = 12;

    dnnl::impl::parallel_nd(n_primitives, [&](memory::dim np) {
        auto relu_d = eltwise_forward::desc(prop_kind::forward_inference,
                algorithm::eltwise_relu, {{np, 1, 1, 1}, dt::f32, tag::nchw},
                0.f, 0.f);
        auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
        auto relu = eltwise_forward(relu_pd);
    });

    ASSERT_EQ(get_primitive_cache_size(), n_primitives);
}

TEST(primitive_cache_mt_test, TestNestedCase) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    engine eng(get_test_engine_kind(), 0);

    // Flush the cache
    set_primitive_cache_capacity(0);
    set_primitive_cache_capacity(1024);

    memory::dim n_primitives = 12;
    memory::dim n_srcs = 32;

    dnnl::impl::parallel_nd(n_primitives, [&](memory::dim np) {
        std::vector<memory::desc> src_mds(n_srcs);
        std::vector<float> scales(n_srcs, 1.0);

        for (memory::dim ns = 0; ns < n_srcs; ++ns) {
            src_mds[ns] = memory::desc({{128, 128}, dt::f32, tag::nc});
        }
        auto sum_pd = sum::primitive_desc(scales, src_mds, eng);
        auto sum_prim = sum(sum_pd);
    });
}

TEST(primitive_cache_mt_test, TestMTCacheHit) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    engine eng(get_test_engine_kind(), 0);

    // Flush the cache
    dnnl::set_primitive_cache_capacity(0);
    dnnl::set_primitive_cache_capacity(1024);

    int n_primitives = 10;

    auto create_eltwise_primitive = [&](int np) {
        auto relu_d = eltwise_forward::desc(prop_kind::forward_inference,
                algorithm::eltwise_relu, {{np, 1, 1, 1}, dt::f32, tag::nchw},
                0.f, 0.f);
        auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
        auto relu = eltwise_forward(relu_pd);
    };

    // Fill the cache with n_primitives (cache_miss)
    for (int i = 0; i < n_primitives; i++)
        create_eltwise_primitive(i);

    // This section should only perform cache_hits
    dnnl::impl::parallel(0, [&](int ithr, int nthr) {
        for (int i = 0; i < n_primitives; i++)
            create_eltwise_primitive(i);
    });

    ASSERT_EQ(get_primitive_cache_size(), n_primitives);
}

} // namespace dnnl
