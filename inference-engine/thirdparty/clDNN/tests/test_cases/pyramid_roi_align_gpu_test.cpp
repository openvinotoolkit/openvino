// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/pyramid_roi_align.hpp>
#include <cldnn/primitives/data.hpp>

using namespace cldnn;
using namespace ::tests;

template <typename T>
struct pyramid_roi_align_typed_test : testing::Test {
    static const data_types data_type = type_to_data_type<T>::value;
    using Type = T;
};
using pyramid_roi_align_types = testing::Types<float, half_t>;

TYPED_TEST_SUITE(pyramid_roi_align_typed_test, pyramid_roi_align_types);

TYPED_TEST(pyramid_roi_align_typed_test, smoke_4levels) {
    using Type = typename pyramid_roi_align_typed_test<TypeParam>::Type;

    auto& engine = get_test_engine();

    const int rois_num = 3;
    const int output_size = 2;
    const int sampling_points = 2;
    const int starting_level = 2;
    const int P2_scale = 1;
    const int P3_scale = 2;
    const int P4_scale = 4;
    const int P5_scale = 8;
    const int P2_size = 8;
    const int P3_size = P2_size * P2_scale / P3_scale;
    const int P4_size = P2_size * P2_scale / P4_scale;
    const int P5_size = P2_size * P2_scale / P5_scale;

    std::vector<Type> rois_data = {
        Type(0.f), Type(0.f), Type(1.f), Type(1.f),
        Type(0.f), Type(0.f), Type(0.5f), Type(0.5f),
        Type(0.5f), Type(0.5f), Type(0.75f), Type(0.75f)
    };

    std::vector<Type> P2_data = {
        Type(0.f), Type(1.f), Type(2.f), Type(3.f), Type(4.f), Type(5.f), Type(6.f), Type(7.f),
        Type(8.f), Type(9.f), Type(10.f), Type(11.f), Type(12.f), Type(13.f), Type(14.f), Type(15.f),
        Type(0.f), Type(1.f), Type(2.f), Type(3.f), Type(4.f), Type(5.f), Type(6.f), Type(7.f),
        Type(8.f), Type(9.f), Type(10.f), Type(11.f), Type(12.f), Type(13.f), Type(14.f), Type(15.f),
        Type(8.f), Type(9.f), Type(10.f), Type(11.f), Type(12.f), Type(13.f), Type(14.f), Type(15.f),
        Type(0.f), Type(1.f), Type(2.f), Type(3.f), Type(4.f), Type(5.f), Type(6.f), Type(7.f),
        Type(0.f), Type(1.f), Type(2.f), Type(3.f), Type(4.f), Type(5.f), Type(6.f), Type(7.f),
        Type(8.f), Type(9.f), Type(10.f), Type(11.f), Type(12.f), Type(13.f), Type(14.f), Type(15.f),
    };

    std::vector<Type> P3_data = {
        Type(9.f), Type(13.f), Type(17.f), Type(21.f),
        Type(9.f), Type(13.f), Type(17.f), Type(21.f),
        Type(9.f), Type(13.f), Type(17.f), Type(21.f),
        Type(9.f), Type(13.f), Type(17.f), Type(21.f),
    };

    std::vector<Type> P4_data = {
      Type(11.f), Type(19.f),
      Type(11.f), Type(19.f),
    };

    std::vector<Type> P5_data = {
        Type(15.f)
    };

    auto rois_lay = layout(this->data_type, format::bfyx, tensor(batch(rois_num), feature(4)));
    auto P2_lay = layout(this->data_type, format::bfyx, tensor(1, 1, P2_size, P2_size));
    auto P3_lay = layout(this->data_type, format::bfyx, tensor(1, 1, P3_size, P3_size));
    auto P4_lay = layout(this->data_type, format::bfyx, tensor(1, 1, P4_size, P4_size));
    auto P5_lay = layout(this->data_type, format::bfyx, tensor(1, 1, P5_size, P5_size));

    auto rois_mem = engine.allocate_memory(rois_lay);
    auto P2_mem = engine.allocate_memory(P2_lay);
    auto P3_mem = engine.allocate_memory(P3_lay);
    auto P4_mem = engine.allocate_memory(P4_lay);
    auto P5_mem = engine.allocate_memory(P5_lay);

    tests::set_values(rois_mem, rois_data);
    tests::set_values(P2_mem, P2_data);
    tests::set_values(P3_mem, P3_data);
    tests::set_values(P4_mem, P4_data);
    tests::set_values(P5_mem, P5_data);

    topology topo;
    topo.add(data("P2", P2_mem));
    topo.add(data("P3", P3_mem));
    topo.add(data("P4", P4_mem));
    topo.add(data("P5", P5_mem));
    topo.add(input_layout("rois", rois_lay));
    topo.add(pyramid_roi_align("pyramid",
                               "rois",
                               "P2",
                               "P3",
                               "P4",
                               "P5",
                               output_size,
                               sampling_points,
                               { P2_scale, P3_scale, P4_scale, P5_scale },
                               starting_level));

    auto net = network(engine, topo);
    net.set_input_data("rois", rois_mem);

    std::vector<float> expected_out = {
        // RoI 0,0 - 1,1 from P4
        14.f, 18.f, 14.f, 18.f,
        // RoI 0,0 - 0.5,0.5 from P3
        11.25f, 14.25f, 11.25f, 14.25f,
        // RoI 0.5,0.5 - 0.75,0.75 from P2
        12.15625f, 13.03125f, 7.40625f, 8.28125f,
    };

    auto result = net.execute();

    auto out_mem = result.at("pyramid").get_memory();
    cldnn::mem_lock<Type> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], static_cast<float>(out_ptr[i])) << "at i = " << i;
    }
}
