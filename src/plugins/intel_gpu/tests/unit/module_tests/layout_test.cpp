// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/layout.hpp"
#include "impls/ocl/kernel_selector_helper.h"

using namespace cldnn;
using namespace ::tests;


struct layout_test_params {
    data_types dt;
    format fmt;
    std::vector<tensor::value_type> size;
    std::vector<tensor::value_type> expected_aligned_size;
    std::vector<size_t> expected_order;
    padding padd;
    std::vector<tensor::value_type> expected_pitches;
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

    auto l = layout(p.dt, p.fmt, tensor{default_fmt, p.size}, p.padd);

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

    ASSERT_EQ(l.get_pitches(), p.expected_pitches);
}

INSTANTIATE_TEST_SUITE_P(smoke, data_layout_test,
    testing::ValuesIn(std::vector<layout_test_params>{
        {data_types::f32, format::bfyx, {1, 15, 5, 5}, {1, 17, 11, 9}, {0, 1, 2, 3}, padding{{0, 1, 3, 2}, 0}/*padding in shape order*/, {1683, 99, 9, 1} /*expected pitches in shape order*/},
        {data_types::f32, format::bfyx, {1, 15, 5, 5}, {1, 15, 5, 5}, {0, 1, 2, 3}, {}, {375, 25, 5, 1}},
        {data_types::f32, format::byxf, {1, 15, 5, 5}, {1, 17, 11, 9}, {0, 2, 3, 1}, padding{{0, 1, 3, 2}, 0}/*padding in shape order*/, {1683, 1, 153, 17} /*expected pitches in shape order*/},
        {data_types::f32, format::byxf, {1, 15, 5, 5}, {1, 15, 5, 5}, {0, 2, 3, 1}, {}, {375, 1, 75, 15}},
        {data_types::f32, format::bfyx, {2, 33, 3, 5}, {2, 33, 3, 5}, {0, 1, 2, 3}, {}, {495, 15, 5, 1}},
        {data_types::f16, format::bfzyx, {2, 33, 3, 5, 4}, {2, 33, 3, 5, 4}, {0, 1, 2, 3, 4}, {}, {1980, 60, 20, 4, 1}},
        {data_types::i8, format::bfwzyx, {2, 33, 3, 5, 4, 6}, {2, 33, 3, 5, 4, 6}, {0, 1, 2, 3, 4, 5}, {}, {11880, 360, 120, 24, 6, 1}},
        {data_types::u8, format::yxfb, {2, 33, 3, 5}, {2, 33, 3, 5}, {2, 3, 1, 0}, {}, {1, 2, 330, 66}},
        {data_types::f32, format::byxf, {2, 33, 3, 5}, {2, 33, 3, 5}, {0, 2, 3, 1}, {}, {495, 1, 165, 33}},
        {data_types::f32, format::fyxb, {2, 33, 3, 5}, {2, 33, 3, 5}, {1, 2, 3, 0}, {}, {1, 30, 10, 2}},
        {data_types::f32, format::b_fs_yx_fsv16, {2, 33, 3, 5}, {2, 48, 3, 5}, {0, 1, 2, 3}, {}, {495, 15, 5, 1}},
        {data_types::f32, format::b_fs_yx_fsv32, {2, 33, 3, 5}, {2, 64, 3, 5}, {0, 1, 2, 3}, {}, {495, 15, 5, 1}},
        {data_types::f32, format::b_fs_zyx_fsv16, {2, 33, 3, 5, 6}, {2, 48, 3, 5, 6}, {0, 1, 2, 3, 4}, {}, {2970, 90, 30, 6, 1}},
        {data_types::f32, format::b_fs_zyx_fsv32, {2, 33, 3, 5, 6}, {2, 64, 3, 5, 6}, {0, 1, 2, 3, 4}, {}, {2970, 90, 30, 6, 1}},
        {data_types::f32, format::bs_fs_zyx_bsv16_fsv16, {2, 33, 3, 5, 6}, {16, 48, 3, 5, 6}, {0, 1, 2, 3, 4}, {}, {2970, 90, 30, 6, 1}},
        {data_types::f32, format::bs_fs_yx_bsv16_fsv16, {2, 33, 3, 5}, {16, 48, 3, 5}, {0, 1, 2, 3}, {}, {495, 15, 5, 1}},
        {data_types::f32, format::bs_fs_yx_bsv4_fsv4, {2, 33, 3, 5}, {4, 36, 3, 5}, {0, 1, 2, 3}, {}, {495, 15, 5, 1}},
        {data_types::f32, format::bfzyx, {3, 2, 2, 2, 2}, {3, 2, 2, 2, 2}, {0, 1, 2, 3, 4}, {}, {16, 8, 4, 2, 1}},
        {data_types::f32, format::bzyxf, {3, 2, 2, 2, 2}, {3, 2, 2, 2, 2}, {0, 2, 3, 4, 1}, {}, {16, 1, 8, 4, 2}},
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

    auto l = layout(p.dt, p.fmt, tensor{default_fmt, p.size}, p.padd);

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

    if (p.expected_pitches.size() > 0)
        ASSERT_EQ(l.get_pitches(), p.expected_pitches);
    else {
        l.get_pitches();
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, weights_layout_test,
    testing::ValuesIn(std::vector<layout_test_params>{
        {data_types::f32, format::oiyx, {2, 15, 3, 5}, {2, 15, 3, 5}, {0, 1, 2, 3}, {}, {225, 15, 5, 1}},
        {data_types::f32, format::ioyx, {2, 15, 3, 5}, {2, 15, 3, 5}, {1, 0, 2, 3}, {}, {15, 30, 5, 1}},
        {data_types::f32, format::yxio, {2, 15, 3, 5}, {2, 15, 3, 5}, {2, 3, 1, 0}, {}, {1, 2, 150, 30}},
        {data_types::f32, format::goiyx, {4, 2, 15, 3, 5}, {4, 2, 15, 3, 5}, {0, 1, 2, 3, 4}, {}, {450, 225, 15, 5, 1}},
        {data_types::f32, format::goizyx, {4, 2, 15, 3, 5, 6}, {4, 2, 15, 3, 5, 6}, {0, 1, 2, 3, 4, 5}, {}, {2700, 1350, 90, 30, 6, 1}},
        {data_types::f32, format::giozyx, {4, 2, 15, 3, 5, 6}, {4, 2, 15, 3, 5, 6}, {0, 2, 1, 3, 4, 5}, {}, {2700, 90, 180, 30, 6, 1}},
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
        {layout{ov::PartialShape{3, 2, 2, 2, 2}, data_types::f16, format::bzyxf},
         layout{ov::PartialShape{3, 2, 2, 2, 2}, data_types::f16, format::bfzyx}, false, false},
        {layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{1, 2, 3, 4, 1, 1}, data_types::f16, format::bfwzyx}, false, true},
        {layout{ov::PartialShape{1, 2, 3, 4, 1, 1}, data_types::f16, format::bfwzyx},
         layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx}, false, true},
        {layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{1, 2, 1, 1, 3, 4}, data_types::f16, format::bfwzyx}, false, true},
        {layout{ov::PartialShape{2, 32, 1, 1}, data_types::f16, format::bfyx, padding({0, 0, 0, 0}, 0)},
         layout{ov::PartialShape{2, 32, 1, 1}, data_types::f16, format::b_fs_yx_fsv16, padding({0, 0, 0, 0}, 0)}, false, true},
        {layout{ov::PartialShape{2, 32, 1, 1}, data_types::f16, format::b_fs_yx_fsv16, padding({0, 0, 0, 0}, 0)},
         layout{ov::PartialShape{2, 32, 1, 1}, data_types::f16, format::b_fs_yx_fsv32, padding({0, 0, 0, 0}, 0)}, false, true},
        {layout{ov::PartialShape{1, 32, 4, 4}, data_types::f32, format::b_fs_yx_fsv32, padding({0, 0, 1, 1}, 0)},
         layout{ov::PartialShape{1, 32, 4, 4}, data_types::f32, format::b_fs_yx_fsv32, padding({0, 0, 0, 0}, 0)}, false, false},
        {layout{ov::PartialShape{1, 32, 4, 4}, data_types::f32, format::b_fs_yx_fsv32, padding({0, 0, 1, 1}, 0)},
         layout{ov::PartialShape{1, 32, 4, 4}, data_types::f32, format::b_fs_yx_fsv32, padding({0, 0, 1, 1}, 0)}, true, true},
        {layout{ov::PartialShape{1, 2, 4, 3}, data_types::f16, format::bfyx, padding({0, 0, 2, 1}, 0)},
         layout{ov::PartialShape{1, 2, 6, 7}, data_types::f16, format::bfyx, padding({0, 0, 0, 0}, 0)}, false, false},
        {layout{ov::PartialShape{10, 20}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{10, 20}, data_types::f16, format::os_iyx_osv16}, false, false},
        {layout{ov::PartialShape{1, 16, 1, 1}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{1, 16, 1, 1}, data_types::f16, format::os_iyx_osv16}, false, false},
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
        {layout{ov::PartialShape{4, 2, 3, 4, 5}, data_types::f16, format::goiyx},
         layout{ov::PartialShape{4, 2, 3, 4, 5}, data_types::f16, format::gioyx}, false, false},
        {layout{ov::PartialShape{4, 1, 16, 16}, data_types::f16, format::bfyx},
         layout{ov::PartialShape{4, 1, 16, 16}, data_types::f16, format::byxf}, false, true},
        {layout{ov::PartialShape{2, 1, 2, 4}, data_types::f16, format::bfyx, padding({0, 0, 1, 0}, {0, 0, 1, 0})},
         layout{ov::PartialShape{2, 1, 2, 4}, data_types::f16, format::bfyx, padding({0, 1, 0, 0}, {0, 0, 0, 0})}, false, false},
    }));

struct layouts_transform_test_params {
    format::type from;
    format::type to;
    ov::PartialShape shape;
    ov::PartialShape expected;
};

class layout_transform_test : public testing::TestWithParam<layouts_transform_test_params> { };

TEST_P(layout_transform_test, basic) {
    auto p = GetParam();

    ASSERT_EQ(layout::transform(p.shape, p.from, p.to), p.expected)
        << "from=" << fmt_to_str(p.from) << " to=" << fmt_to_str(p.to) << " shape=" << p.shape;
}

INSTANTIATE_TEST_SUITE_P(smoke, layout_transform_test,
    testing::ValuesIn(std::vector<layouts_transform_test_params>{
        {format::yxfb, format::bfyx, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{4, 3, 1, 2}},
        {format::bfyx, format::yxfb, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{3, 4, 2, 1}},
        {format::bfyx, format::bs_f_bsv16, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{1, 2*3*4}},
        {format::bs_f_bsv16, format::bfyx, ov::PartialShape{1, 2*3*4}, ov::PartialShape{1, 2*3*4, 1, 1}},
        {format::bfyx, format::bs_fs_yx_bsv16_fsv16, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{1, 2, 3, 4}},
        {format::bfyx, format::bfzyx, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{1, 2, 1, 3, 4}},
        {format::bfyx, format::bfwzyx, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{1, 2, 1, 1, 3, 4}},
        {format::bfyx, format::bfuwzyx, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{1, 2, 1, 1, 1, 3, 4}},
        {format::bfyx, format::bfvuwzyx, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{1, 2, 1, 1, 1, 1, 3, 4}},
        {format::bfyx, format::bfvuwzyx, ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 1, 1, 1, 1, 3, 1}},

        {format::b_fs_yx_fsv16, format::bfyx, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{1, 2, 3, 4}},
        {format::b_fs_yx_fsv16, format::bfzyx, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{1, 2, 1, 3, 4}},
        {format::b_fs_yx_fsv16, format::bfwzyx, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{1, 2, 1, 1, 3, 4}},
        {format::b_fs_yx_fsv16, format::bfuwzyx, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{1, 2, 1, 1, 1, 3, 4}},
        {format::b_fs_yx_fsv16, format::bfvuwzyx, ov::PartialShape{1, 2, 3, 4}, ov::PartialShape{1, 2, 1, 1, 1, 1, 3, 4}},

        {format::bfzyx, format::b_fs_zyx_fsv16, ov::PartialShape{1, 2, 3, 4, 5}, ov::PartialShape{1, 2, 3, 4, 5}},
        {format::bfzyx, format::bfwzyx, ov::PartialShape{1, 2, 3, 4, 5}, ov::PartialShape{1, 2, 1, 3, 4, 5}},
        {format::bfzyx, format::bfuwzyx, ov::PartialShape{1, 2, 3, 4, 5}, ov::PartialShape{1, 2, 1, 1, 3, 4, 5}},
        {format::bfzyx, format::bfvuwzyx, ov::PartialShape{1, 2, 3, 4, 5}, ov::PartialShape{1, 2, 1, 1, 1, 3, 4, 5}},

        {format::b_fs_zyx_fsv16, format::bfzyx, ov::PartialShape{1, 2, 3, 4, 5}, ov::PartialShape{1, 2, 3, 4, 5}},
        {format::b_fs_zyx_fsv16, format::bfwzyx, ov::PartialShape{1, 2, 3, 4, 5}, ov::PartialShape{1, 2, 1, 3, 4, 5}},
        {format::b_fs_zyx_fsv16, format::bfuwzyx, ov::PartialShape{1, 2, 3, 4, 5}, ov::PartialShape{1, 2, 1, 1, 3, 4, 5}},
        {format::b_fs_zyx_fsv16, format::bfvuwzyx, ov::PartialShape{1, 2, 3, 4, 5}, ov::PartialShape{1, 2, 1, 1, 1, 3, 4, 5}},

        {format::bfwzyx, format::bfuwzyx, ov::PartialShape{1, 2, 3, 4, 5, 6}, ov::PartialShape{1, 2, 1, 3, 4, 5, 6}},
        {format::bfwzyx, format::bfvuwzyx, ov::PartialShape{1, 2, 3, 4, 5, 6}, ov::PartialShape{1, 2, 1, 1, 3, 4, 5, 6}},

        {format::bfuwzyx, format::bfvuwzyx, ov::PartialShape{1, 2, 3, 4, 5, 6, 7}, ov::PartialShape{1, 2, 1, 3, 4, 5, 6, 7}},

        {format::bfvuwzyx, format::bfuwzyx,  ov::PartialShape{1, 2, 3, 4, 5, 6, 7, 8}, ov::PartialShape{1, 2, 3*4, 5, 6, 7, 8}},
        {format::bfvuwzyx, format::bfuwzyx,  ov::PartialShape{1, 2, 3, 4, 5, 6, 7, 8}, ov::PartialShape{1, 2, 3*4, 5, 6, 7, 8}},
        {format::bfvuwzyx, format::bfzyx,  ov::PartialShape{1, 2, 3, 4, 5, 6, 7, 8}, ov::PartialShape{1, 2, 3*4*5*6, 7, 8}},
        {format::bfvuwzyx, format::bfyx,  ov::PartialShape{1, 2, 3, 4, 5, 6, 7, 8}, ov::PartialShape{1, 2, 3*4*5*6*7, 8}},

        {format::bfuwzyx, format::bfwzyx,  ov::PartialShape{1, 2, 3, 4, 5, 6, 7}, ov::PartialShape{1, 2, 3*4, 5, 6, 7}},
        {format::bfuwzyx, format::bfzyx,  ov::PartialShape{1, 2, 3, 4, 5, 6, 7}, ov::PartialShape{1, 2, 3*4*5, 6, 7}},
        {format::bfuwzyx, format::bfyx,  ov::PartialShape{1, 2, 3, 4, 5, 6, 7}, ov::PartialShape{1, 2, 3*4*5*6, 7}},

        {format::bfwzyx, format::bfzyx,  ov::PartialShape{1, 2, 3, 4, 5, 6}, ov::PartialShape{1, 2, 3*4, 5, 6}},
        {format::bfwzyx, format::bfyx,  ov::PartialShape{1, 2, 3, 4, 5, 6}, ov::PartialShape{1, 2, 3*4*5, 6}},

        {format::bfzyx, format::bfyx,  ov::PartialShape{1, 2, 3, 4, 5}, ov::PartialShape{1, 2, 3*4, 5}},
    }));

struct layouts_convert_params {
    format::type in_format;
    ov::PartialShape in_shape;
    bool is_grouped;
};

class layout_convert_test : public testing::TestWithParam<layouts_convert_params> { };

TEST_P(layout_convert_test, basic) {
    auto p = GetParam();

    auto test_layout = layout(p.in_shape, data_types::f32, p.in_format);
    auto weights_tensor = convert_weights_tensor(test_layout, p.is_grouped);
    auto converted_layout = from_weights_tensor(weights_tensor);

    if (p.in_format == format::bfzyx && p.is_grouped) {
        ASSERT_EQ(converted_layout, layout(p.in_shape, data_types::f32, format::goiyx));
    } else if (p.in_format == format::bfwzyx && p.is_grouped) {
        ASSERT_EQ(converted_layout, layout(p.in_shape, data_types::f32, format::goizyx));
    } else if (p.in_format == format::os_i_osv16__ai8) {
        auto ref_shape = p.in_shape;
        for (size_t i = ref_shape.size(); i < converted_layout.get_dims().size(); ++i)
            ref_shape.push_back(1);
        test_layout.set_partial_shape(ref_shape);
        ASSERT_EQ(test_layout, converted_layout);
    } else {
        ASSERT_EQ(test_layout, converted_layout);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, layout_convert_test,
    testing::ValuesIn(std::vector<layouts_convert_params>{
        // 4D formats
        {format::oiyx, ov::PartialShape{1, 2, 3, 4}, false},
        {format::ioyx, ov::PartialShape{1, 2, 3, 4}, false},
        {format::os_i_osv16__ai8, ov::PartialShape{1, 2}, false},
        {format::os_iyx_osv16, ov::PartialShape{1, 2, 3, 4}, false},
        // 4D formats grouped
        {format::bfzyx, ov::PartialShape{1, 2, 3, 4, 5}, true},
        {format::goiyx, ov::PartialShape{1, 2, 3, 4, 5}, false},
        {format::g_os_iyx_osv32, ov::PartialShape{1, 2, 3, 4, 5}, false},
        {format::g_os_is_yx_isv8_osv16_isv2, ov::PartialShape{1, 2, 3, 4, 5}, false},
        {format::g_os_is_yx_osv16_isv4, ov::PartialShape{1, 2, 3, 4, 5}, false},
        // {format::gs_oi_yxs_gsv32_yxsv4, ov::PartialShape{1, 2, 3, 4, 5}, false},
        // 5D formats
        {format::oizyx, ov::PartialShape{1, 2, 3, 4, 5}, false},
        {format::iozyx, ov::PartialShape{1, 2, 3, 4, 5}, false},
        {format::os_is_zyx_isa8_osv16_isv4, ov::PartialShape{1, 2, 3, 4, 5}, false},
        {format::os_is_zyx_osa4_isa8_osv8_isv4, ov::PartialShape{1, 2, 3, 4, 5}, false},
        {format::is_os_zyx_isv16_osv16, ov::PartialShape{1, 2, 3, 4, 5}, false},
        // 5D formats grouped
        {format::bfwzyx, ov::PartialShape{1, 2, 3, 4, 5, 6}, true},
        {format::giozyx, ov::PartialShape{1, 2, 3, 4, 5, 6}, false},
        {format::g_os_zyx_is_osv32_isv32, ov::PartialShape{1, 2, 3, 4, 5, 6}, false},
        {format::g_is_os_zyx_isv16_osv16, ov::PartialShape{1, 2, 3, 4, 5, 6}, false},
    }));

struct custom_layout_test_params {
    ov::PartialShape shape;
    cldnn::format_traits left;
    cldnn::format_traits right;
};

class custom_layout_test : public testing::TestWithParam<custom_layout_test_params> { };

TEST_P(custom_layout_test, different_hash) {
    auto p = GetParam();
    auto left = cldnn::layout(p.shape, cldnn::data_types::f16, cldnn::format(p.left));
    auto right = cldnn::layout(p.shape, cldnn::data_types::f16, cldnn::format(p.right));
    ASSERT_TRUE(left.hash() != right.hash());
}

TEST_P(custom_layout_test, same_hash) {
    auto p = GetParam();
    auto left = cldnn::layout(p.shape, cldnn::data_types::f16, cldnn::format(p.left));
    auto right = cldnn::layout(p.shape, cldnn::data_types::f16, cldnn::format(p.left));
    ASSERT_TRUE(left.hash() == right.hash());

    left = cldnn::layout(p.shape, cldnn::data_types::f16, cldnn::format(p.right));
    right = cldnn::layout(p.shape, cldnn::data_types::f16, cldnn::format(p.right));
    ASSERT_TRUE(left.hash() == right.hash());
}

INSTANTIATE_TEST_SUITE_P(smoke, custom_layout_test,
    testing::ValuesIn(std::vector<custom_layout_test_params>{
        {
            {16, 16, 8, 8},
            format_traits{
                "custom", 1, 1, 2, 0, {0, 1, 2, 3}, "oiyx", "oixy?", {{1, 16}, {0, 16}}
            },
            format_traits{
                "custom", 1, 1, 2, 0, {0, 1, 2, 3}, "oiyx", "oixy?", {{0, 2}, {1, 8}, {0, 8}, {1, 2}}
            }
        },
        {
            {32, 32, 8, 8},
            format_traits{
                "custom", 1, 1, 2, 0, {0, 1, 2, 3}, "oiyx", "oixy?", {{1, 4}, {0, 8}, {1, 8}, {0, 4}}
            },
            format_traits{
                "custom", 1, 1, 2, 0, {0, 1, 2, 3}, "oiyx", "oixy?", {{0, 2}, {1, 8}, {0, 8}, {1, 2}}
            }
        },
    }));
