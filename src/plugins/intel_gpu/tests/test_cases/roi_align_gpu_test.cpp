// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/roi_align.hpp>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

template <typename TD, typename TI, cldnn::format::type fmt>
struct TypesWithFormat {
    using DataType = TD;
    using IndexType = TI;
    static const cldnn::format::type format = fmt;
};

template <typename Types>
struct roi_align_test : public testing::Test {
    using TD = typename Types::DataType;
    using TI = typename Types::IndexType;
    const data_types device_data_type = type_to_data_type<TD>::value;
    const data_types device_ind_type = type_to_data_type<TI>::value;
    const cldnn::format::type blocked_format = Types::format;
    const cldnn::format::type plain_format = format::bfyx;

    const int pooled_h{2};
    const int pooled_w{2};
    const int sampling_ratio{2};
    const float spatial_scale{1};

    const std::vector<TD> input_data = {
        TD(0.f),  TD(1.f),  TD(8.f),  TD(5.f),  TD(5.f), TD(2.f),  TD(0.f), TD(7.f),  TD(7.f),  TD(10.f), TD(4.f),
        TD(5.f),  TD(9.f),  TD(0.f),  TD(0.f),  TD(5.f), TD(7.f),  TD(0.f), TD(4.f),  TD(0.f),  TD(4.f),  TD(7.f),
        TD(6.f),  TD(10.f), TD(9.f),  TD(5.f),  TD(1.f), TD(7.f),  TD(4.f), TD(7.f),  TD(10.f), TD(8.f),  TD(2.f),
        TD(0.f),  TD(8.f),  TD(3.f),  TD(6.f),  TD(8.f), TD(10.f), TD(4.f), TD(2.f),  TD(10.f), TD(7.f),  TD(8.f),
        TD(7.f),  TD(0.f),  TD(6.f),  TD(9.f),  TD(2.f), TD(4.f),  TD(8.f), TD(5.f),  TD(2.f),  TD(3.f),  TD(3.f),
        TD(1.f),  TD(5.f),  TD(9.f),  TD(10.f), TD(0.f), TD(9.f),  TD(5.f), TD(5.f),  TD(3.f),  TD(10.f), TD(5.f),
        TD(2.f),  TD(0.f),  TD(10.f), TD(0.f),  TD(5.f), TD(4.f),  TD(3.f), TD(10.f), TD(5.f),  TD(5.f),  TD(10.f),
        TD(0.f),  TD(8.f),  TD(8.f),  TD(9.f),  TD(1.f), TD(0.f),  TD(7.f), TD(9.f),  TD(6.f),  TD(8.f),  TD(7.f),
        TD(10.f), TD(9.f),  TD(2.f),  TD(3.f),  TD(3.f), TD(5.f),  TD(6.f), TD(9.f),  TD(4.f),  TD(9.f),  TD(2.f),
        TD(4.f),  TD(5.f),  TD(5.f),  TD(3.f),  TD(1.f), TD(1.f),  TD(6.f), TD(8.f),  TD(0.f),  TD(5.f),  TD(5.f),
        TD(10.f), TD(8.f),  TD(6.f),  TD(9.f),  TD(6.f), TD(9.f),  TD(1.f), TD(2.f),  TD(7.f),  TD(1.f),  TD(1.f),
        TD(3.f),  TD(0.f),  TD(4.f),  TD(0.f),  TD(7.f), TD(10.f), TD(2.f)};
    const std::vector<TD> coords_data = {TD(2.f), TD(2.f), TD(4.f), TD(4.f), TD(2.f), TD(2.f), TD(4.f), TD(4.f)};
    const std::vector<TI> roi_data = {0, 1};

    const layout input_lt = layout(device_data_type, plain_format, {2, 1, 8, 8});
    const layout coords_lt = layout(device_data_type, plain_format, {2, 4, 1, 1});
    const layout roi_lt = layout(device_ind_type, plain_format, {2, 1, 1, 1});

    memory::ptr get_memory(engine& engine, const layout& lt, const std::vector<TD>& data) const {
        auto mem = engine.allocate_memory(lt);
        tests::set_values(mem, data);
        return mem;
    }

    memory::ptr get_roi_memory(engine& engine) const {
        auto mem = engine.allocate_memory(roi_lt);
        tests::set_values(mem, roi_data);
        return mem;
    }

    void execute(const std::vector<TD>& expected_output,
                 roi_align::PoolingMode pooling_mode,
                 roi_align::AlignedMode aligned_mode) const {
        auto& engine = get_test_engine();

        auto input = get_memory(engine, input_lt, input_data);
        auto coords = get_memory(engine, coords_lt, coords_data);
        auto roi_ind = get_roi_memory(engine);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(input_layout("coords", coords->get_layout()));
        topology.add(input_layout("roi_ind", roi_ind->get_layout()));
        topology.add(reorder("reorder_input", input_info("input"), blocked_format, device_data_type));
        topology.add(reorder("reorder_coords", input_info("coords"), blocked_format, device_data_type));
        topology.add(reorder("reorder_ind", input_info("roi_ind"), blocked_format, device_ind_type));
        topology.add(roi_align("roi_align",
                               { input_info("reorder_input"), input_info("reorder_coords"), input_info("reorder_ind") },
                               pooled_h,
                               pooled_w,
                               sampling_ratio,
                               spatial_scale,
                               pooling_mode,
                               aligned_mode));
        topology.add(reorder("out", input_info("roi_align"), plain_format, device_data_type));

        network network(engine, topology);
        network.set_input_data("input", input);
        network.set_input_data("coords", coords);
        network.set_input_data("roi_ind", roi_ind);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<TD> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), expected_output.size());
        for (uint32_t i = 0; i < expected_output.size(); ++i) {
            ASSERT_NEAR(output_ptr[i], expected_output[i], 0.01);
        }
    }
};

// it's a bit overloaded with the cartesian product of types and formats, but that's the lesser evil
// since we have specific type for expected values that are tied to specific input modes
// so that Combine approach could avoid manual combinations but it would be much more complicated
using roi_align_test_types = testing::Types<TypesWithFormat<half_t, uint8_t, format::bfyx>,
                                            TypesWithFormat<half_t, uint8_t, format::b_fs_yx_fsv16>,
                                            TypesWithFormat<half_t, uint8_t, format::b_fs_yx_fsv32>,
                                            TypesWithFormat<half_t, uint8_t, format::bs_fs_yx_bsv16_fsv16>,
                                            TypesWithFormat<half_t, uint8_t, format::bs_fs_yx_bsv32_fsv16>,
                                            TypesWithFormat<half_t, uint8_t, format::bs_fs_yx_bsv32_fsv32>,

                                            TypesWithFormat<half_t, int8_t, format::bfyx>,
                                            TypesWithFormat<half_t, int8_t, format::b_fs_yx_fsv16>,
                                            TypesWithFormat<half_t, int8_t, format::b_fs_yx_fsv32>,
                                            TypesWithFormat<half_t, int8_t, format::bs_fs_yx_bsv16_fsv16>,
                                            TypesWithFormat<half_t, int8_t, format::bs_fs_yx_bsv32_fsv16>,
                                            TypesWithFormat<half_t, int8_t, format::bs_fs_yx_bsv32_fsv32>,

                                            TypesWithFormat<half_t, int32_t, format::bfyx>,
                                            TypesWithFormat<half_t, int32_t, format::b_fs_yx_fsv16>,
                                            TypesWithFormat<half_t, int32_t, format::b_fs_yx_fsv32>,
                                            TypesWithFormat<half_t, int32_t, format::bs_fs_yx_bsv16_fsv16>,
                                            TypesWithFormat<half_t, int32_t, format::bs_fs_yx_bsv32_fsv16>,
                                            TypesWithFormat<half_t, int32_t, format::bs_fs_yx_bsv32_fsv32>,

                                            TypesWithFormat<float, uint8_t, format::bfyx>,
                                            TypesWithFormat<float, uint8_t, format::b_fs_yx_fsv16>,
                                            TypesWithFormat<float, uint8_t, format::b_fs_yx_fsv32>,
                                            TypesWithFormat<float, uint8_t, format::bs_fs_yx_bsv16_fsv16>,
                                            TypesWithFormat<float, uint8_t, format::bs_fs_yx_bsv32_fsv16>,
                                            TypesWithFormat<float, uint8_t, format::bs_fs_yx_bsv32_fsv32>,

                                            TypesWithFormat<float, int8_t, format::bfyx>,
                                            TypesWithFormat<float, int8_t, format::b_fs_yx_fsv16>,
                                            TypesWithFormat<float, int8_t, format::b_fs_yx_fsv32>,
                                            TypesWithFormat<float, int8_t, format::bs_fs_yx_bsv16_fsv16>,
                                            TypesWithFormat<float, int8_t, format::bs_fs_yx_bsv32_fsv16>,
                                            TypesWithFormat<float, int8_t, format::bs_fs_yx_bsv32_fsv32>,

                                            TypesWithFormat<float, int32_t, format::bfyx>,
                                            TypesWithFormat<float, int32_t, format::b_fs_yx_fsv16>,
                                            TypesWithFormat<float, int32_t, format::b_fs_yx_fsv32>,
                                            TypesWithFormat<float, int32_t, format::bs_fs_yx_bsv16_fsv16>,
                                            TypesWithFormat<float, int32_t, format::bs_fs_yx_bsv32_fsv16>,
                                            TypesWithFormat<float, int32_t, format::bs_fs_yx_bsv32_fsv32>>;

TYPED_TEST_SUITE(roi_align_test, roi_align_test_types);

TYPED_TEST(roi_align_test, avg_asymmetric) {
    using TD = typename TypeParam::DataType;
    const std::vector<TD>
        expected_output{TD(3.f), TD(3.75f), TD(4.75f), TD(5.f), TD(3.f), TD(5.5f), TD(2.75f), TD(3.75f)};
    this->execute(expected_output, roi_align::PoolingMode::avg, roi_align::AlignedMode::asymmetric);
}

TYPED_TEST(roi_align_test, avg_half_pixel_for_nn) {
    using TD = typename TypeParam::DataType;
    const std::vector<TD> expected_output =
        {TD(3.14f), TD(2.16f), TD(2.86f), TD(5.03f), TD(1.83f), TD(5.84f), TD(2.77f), TD(3.44f)};
    this->execute(expected_output, roi_align::PoolingMode::avg, roi_align::AlignedMode::half_pixel_for_nn);
}

TYPED_TEST(roi_align_test, max_half_pixel) {
    using TD = typename TypeParam::DataType;
    const std::vector<TD> expected_output =
        {TD(4.375f), TD(4.9375f), TD(5.6875f), TD(5.625f), TD(4.625f), TD(7.125f), TD(3.3125f), TD(4.3125f)};
    this->execute(expected_output, roi_align::PoolingMode::max, roi_align::AlignedMode::half_pixel);
}
