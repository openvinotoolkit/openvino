// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/layout.hpp"

using namespace cldnn;
using namespace ::tests;


struct layout_test_params {
    data_types dt;
    format fmt;
    std::vector<tensor::value_type> size;
    std::vector<tensor::value_type> expected_aligned_size;
    std::vector<size_t> expected_order;
};

class data_layout_test : public testing::TestWithParam<layout_test_params> { };

TEST_P(data_layout_test, size_check) {
    auto p = GetParam();
    auto default_fmt = format::bfyx;

    if (p.size.size() == 5) {
        default_fmt = format::bfzyx;
    } else if (p.size.size() == 6) {
        default_fmt = format::bfwzyx;
    }

    ASSERT_FALSE(format::is_weights_format(p.fmt));

    auto l = layout(p.dt, p.fmt, tensor{default_fmt, p.size});

    size_t expected_count = std::accumulate(p.size.begin(), p.size.end(), 1, std::multiplies<int>());
    size_t expected_bytes_count = std::accumulate(p.expected_aligned_size.begin(), p.expected_aligned_size.end(), 1, std::multiplies<int>()) *
                                  data_type_traits::size_of(p.dt);

    ASSERT_EQ(l.bytes_count(), expected_bytes_count);
    ASSERT_EQ(l.count(), expected_count);
    ASSERT_EQ(l.get_rank(), p.size.size());

    ASSERT_EQ(l.batch(), p.size[0]);
    ASSERT_EQ(l.feature(), p.size[1]);
    if (p.size.size() == 6) {
        ASSERT_EQ(l.spatial(0), p.size[5]);
        ASSERT_EQ(l.spatial(1), p.size[4]);
        ASSERT_EQ(l.spatial(2), p.size[3]);
        ASSERT_EQ(l.spatial(3), p.size[2]);
    } else if (p.size.size() == 5) {
        ASSERT_EQ(l.spatial(0), p.size[4]);
        ASSERT_EQ(l.spatial(1), p.size[3]);
        ASSERT_EQ(l.spatial(2), p.size[2]);
        ASSERT_EQ(l.spatial(3), 1);
    } else if (p.size.size() == 4) {
        ASSERT_EQ(l.spatial(0), p.size[3]);
        ASSERT_EQ(l.spatial(1), p.size[2]);
        ASSERT_EQ(l.spatial(2), 1);
        ASSERT_EQ(l.spatial(3), 1);
    }

    auto dims = l.get_dims();
    auto ordered_dims = l.get_ordered_dims();

    ASSERT_EQ(dims, p.size);
    ASSERT_EQ(l.get_dims_order(), p.expected_order);
    ASSERT_EQ(l.get_dims_order().size(), dims.size());

    for (auto& dim_idx : l.get_dims_order()) {
        ASSERT_LT(dim_idx, ordered_dims.size());
    }

    for (size_t i = 0; i < l.get_rank(); i++) {
        ASSERT_EQ(ordered_dims[i], dims[p.expected_order[i]]);
        ASSERT_EQ(ordered_dims[i], p.size[p.expected_order[i]]);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, data_layout_test,
    testing::ValuesIn(std::vector<layout_test_params>{
        {data_types::f32, format::bfyx, {2, 33, 3, 5}, {2, 33, 3, 5}, {0, 1, 2, 3}},
        {data_types::f16, format::bfzyx, {2, 33, 3, 5, 4}, {2, 33, 3, 5, 4}, {0, 1, 2, 3, 4}},
        {data_types::i8, format::bfwzyx, {2, 33, 3, 5, 4, 6}, {2, 33, 3, 5, 4, 6}, {0, 1, 2, 3, 4, 5}},
        {data_types::u8, format::yxfb, {2, 33, 3, 5}, {2, 33, 3, 5}, {2, 3, 1, 0}},
        {data_types::f32, format::byxf, {2, 33, 3, 5}, {2, 33, 3, 5}, {0, 2, 3, 1}},
        {data_types::f32, format::fyxb, {2, 33, 3, 5}, {2, 33, 3, 5}, {1, 2, 3, 0}},
        {data_types::f32, format::b_fs_yx_fsv16, {2, 33, 3, 5}, {2, 48, 3, 5}, {0, 1, 2, 3}},
        {data_types::f32, format::b_fs_yx_fsv32, {2, 33, 3, 5}, {2, 64, 3, 5}, {0, 1, 2, 3}},
        {data_types::f32, format::b_fs_zyx_fsv16, {2, 33, 3, 5, 6}, {2, 48, 3, 5, 6}, {0, 1, 2, 3, 4}},
        {data_types::f32, format::b_fs_zyx_fsv32, {2, 33, 3, 5, 6}, {2, 64, 3, 5, 6}, {0, 1, 2, 3, 4}},
        {data_types::f32, format::bs_fs_zyx_bsv16_fsv16, {2, 33, 3, 5, 6}, {16, 48, 3, 5, 6}, {0, 1, 2, 3, 4}},
        {data_types::f32, format::bs_fs_yx_bsv16_fsv16, {2, 33, 3, 5}, {16, 48, 3, 5}, {0, 1, 2, 3}},
        {data_types::f32, format::bs_fs_yx_bsv4_fsv4, {2, 33, 3, 5}, {4, 36, 3, 5}, {0, 1, 2, 3}},
    }));

class weights_layout_test : public testing::TestWithParam<layout_test_params> { };

TEST_P(weights_layout_test, size_check) {
    auto p = GetParam();
    auto default_fmt = format::oiyx;

    if (format::is_weights_format(p.fmt)) {
        if (p.size.size() == 5) {
            default_fmt = format::goiyx;
        } else if (p.size.size() == 6) {
            default_fmt = format::goizyx;
        }
    } else {
        if (p.size.size() == 4) {
            default_fmt = format::oiyx;
        } else if (p.size.size() == 5) {
            default_fmt = format::oizyx;
        }
    }

    auto l = layout(p.dt, p.fmt, tensor{default_fmt, p.size});

    size_t expected_count = std::accumulate(p.size.begin(), p.size.end(), 1, std::multiplies<tensor::value_type>());
    size_t expected_bytes_count = std::accumulate(p.expected_aligned_size.begin(),
                                                  p.expected_aligned_size.end(),
                                                  1,
                                                  std::multiplies<tensor::value_type>()) *
                                  data_type_traits::size_of(p.dt);

    ASSERT_EQ(l.bytes_count(), expected_bytes_count);
    ASSERT_EQ(l.count(), expected_count);
    ASSERT_EQ(l.get_rank(), p.size.size());

    if (format::is_weights_format(p.fmt)) {
        if (format::is_grouped(p.fmt)) {
            ASSERT_EQ(l.group(), p.size[0]);
            ASSERT_EQ(l.ofm(), p.size[1]);
            ASSERT_EQ(l.ifm(), p.size[2]);
            if (p.size.size() == 6) {
                ASSERT_EQ(l.spatial(0), p.size[5]);
                ASSERT_EQ(l.spatial(1), p.size[4]);
                ASSERT_EQ(l.spatial(2), p.size[3]);
            } else if (p.size.size() == 5) {
                ASSERT_EQ(l.spatial(0), p.size[4]);
                ASSERT_EQ(l.spatial(1), p.size[3]);
            }
        } else {
            ASSERT_EQ(l.ofm(), p.size[0]);
            ASSERT_EQ(l.ifm(), p.size[1]);
            if (p.size.size() == 6) {
                ASSERT_EQ(l.spatial(0), p.size[4]);
                ASSERT_EQ(l.spatial(1), p.size[3]);
                ASSERT_EQ(l.spatial(2), p.size[2]);
            } else if (p.size.size() == 5) {
                ASSERT_EQ(l.spatial(0), p.size[3]);
                ASSERT_EQ(l.spatial(1), p.size[2]);
            }
        }
    }
    auto dims = l.get_dims();
    auto ordered_dims = l.get_ordered_dims();

    ASSERT_EQ(dims, p.size);
    ASSERT_EQ(l.get_dims_order(), p.expected_order);
    ASSERT_EQ(l.get_dims_order().size(), dims.size());

    for (auto& dim_idx : l.get_dims_order()) {
        ASSERT_LT(dim_idx, ordered_dims.size());
    }

    for (size_t i = 0; i < l.get_rank(); i++) {
        ASSERT_EQ(ordered_dims[i], dims[p.expected_order[i]]);
        ASSERT_EQ(ordered_dims[i], p.size[p.expected_order[i]]);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, weights_layout_test,
    testing::ValuesIn(std::vector<layout_test_params>{
        {data_types::f32, format::oiyx, {2, 15, 3, 5}, {2, 15, 3, 5}, {0, 1, 2, 3}},
        {data_types::f32, format::ioyx, {2, 15, 3, 5}, {2, 15, 3, 5}, {1, 0, 2, 3}},
        {data_types::f32, format::yxio, {2, 15, 3, 5}, {2, 15, 3, 5}, {2, 3, 1, 0}},
        {data_types::f32, format::goiyx, {4, 2, 15, 3, 5}, {4, 2, 15, 3, 5}, {0, 1, 2, 3, 4}},
        {data_types::f32, format::goizyx, {4, 2, 15, 3, 5, 6}, {4, 2, 15, 3, 5, 6}, {0, 1, 2, 3, 4, 5}},
        {data_types::f32, format::giozyx, {4, 2, 15, 3, 5, 6}, {4, 2, 15, 3, 5, 6}, {0, 2, 1, 3, 4, 5}},
        {data_types::f32, format::g_os_is_yx_osa2_isa8_osv16_isv2, {4, 2, 15, 3, 5}, {4, 32, 16, 3, 5}, {0, 1, 2, 3, 4}},
        {data_types::f32, format::g_os_is_zyx_osa4_isa8_osv8_isv4, {4, 2, 15, 3, 5, 6}, {4, 32, 32, 3, 5, 6}, {0, 1, 2, 3, 4, 5}},
    }));


struct layouts_cmp_test_params {
    layout l1;
    layout l2;
    bool is_identical;
    bool is_compatible;
};

class layout_cmp_test : public testing::TestWithParam<layouts_cmp_test_params> { };

TEST_P(layout_cmp_test, basic) {
    auto p = GetParam();

    EXPECT_EQ(p.l1.identical(p.l2), p.is_identical) << p.l1.to_short_string() << " -> " << p.l2.to_short_string();
    EXPECT_EQ(p.l1.compatible(p.l2), p.is_compatible) << p.l1.to_short_string() << " -> " << p.l2.to_short_string();
}

INSTANTIATE_TEST_SUITE_P(smoke, layout_cmp_test,
    testing::ValuesIn(std::vector<layouts_cmp_test_params>{
        {layout{ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx},
         layout{ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx}, true, true},
        {layout{ov::PartialShape{4, 3, 2, 1}, data_types::f32, format::bfyx},
         layout{ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx}, false, true},
        {layout{ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx},
         layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx}, false, false},
        {layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{1, 2, 1, 3, 4}, data_types::f16, format::bfzyx}, false, true},
        {layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{1, 2, 3, 4, 1, 1}, data_types::f16, format::bfwzyx}, false, true},
        {layout{ov::PartialShape{1, 2, 3, 4, 1, 1}, data_types::f16, format::bfwzyx},
         layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx}, false, true},
        {layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{1, 2, 1, 1, 3, 4}, data_types::f16, format::bfwzyx}, false, true},
        {layout{ov::PartialShape{1, 32, 4, 4}, data_types::f32, format::b_fs_yx_fsv32, padding({0, 0, 1, 1}, 0)},
         layout{ov::PartialShape{1, 32, 4, 4}, data_types::f32, format::b_fs_yx_fsv32, padding({0, 0, 0, 0}, 0)}, false, false},
        {layout{ov::PartialShape{1, 32, 4, 4}, data_types::f32, format::b_fs_yx_fsv32, padding({0, 0, 1, 1}, 0)},
         layout{ov::PartialShape{1, 32, 4, 4}, data_types::f32, format::b_fs_yx_fsv32, padding({0, 0, 1, 1}, 0)}, true, true},
        {layout{ov::PartialShape{10, 20}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{10, 20}, data_types::f16, format::os_iyx_osv16}, false, false},
        {layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::oiyx}, false, true},
        {layout{ov::PartialShape{128, 10}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{128, 10}, data_types::f16, format::os_iyx_osv32}, false, false},
        {layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::yxfb}, false, false},
        {layout{ov::PartialShape{1, 2, 1, 1}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{1, 2, 1, 1}, data_types::f16, format::b_fs_yx_fsv16}, false, false},
        {layout{ov::PartialShape{1, 2, 1, 1, 1}, data_types::f16, format::b_fs_zyx_fsv16},
         layout{ov::PartialShape{1, 2, 1, 1}, data_types::f16, format::b_fs_yx_fsv16}, false, false},
        {layout{ov::PartialShape{4, 2, 3, 4, 5}, data_types::f16, format::os_is_zyx_isv16_osv16},
         layout{ov::PartialShape{4, 2, 3, 4, 5}, data_types::f16, format::is_os_zyx_isv16_osv16}, false, false},
        {layout{ov::PartialShape{4, 2, 3, 4, 5}, data_types::f16, format::g_os_yx_is_osv8_isv2},
         layout{ov::PartialShape{4, 2, 3, 4, 5}, data_types::f16, format::g_os_y_is_x_osv8_isv2}, false, false},
        {layout{ov::PartialShape{4, 2, 3, 4, 5}, data_types::f16, format::goiyx},
         layout{ov::PartialShape{4, 2, 3, 4, 5}, data_types::f16, format::gioyx}, false, false},
        {layout{ov::PartialShape{9, 17, 3, 2, 5}, data_types::f16, format::is_os_zyx_isa8_osv8_isv2},
         layout{ov::PartialShape{9, 17, 3, 2, 5}, data_types::f16, format::os_is_zyx_isa8_osv8_isv2}, false, false},
    }));
