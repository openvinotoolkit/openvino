// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/engine.hpp>
#include <api/input_layout.hpp>
#include <api/pyramid_roi_align.hpp>
#include <api/memory.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/mutable_data.hpp>
#include <api/data.hpp>

#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

template <typename T>
struct pyramid_roi_align_typed_test : testing::Test {
    static const data_types data_type = type_to_data_type<T>::value;
    using Type = T;
};
using pyramid_roi_align_types = testing::Types<float, half_t>;

TYPED_TEST_CASE(pyramid_roi_align_typed_test, pyramid_roi_align_types);

TYPED_TEST(pyramid_roi_align_typed_test, smoke_4levels) {
    using Type = typename pyramid_roi_align_typed_test<TypeParam>::Type;

    const auto& engine = get_test_engine();

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

    auto rois_mem = memory::allocate(engine, rois_lay);
    auto P2_mem = memory::allocate(engine, P2_lay);
    auto P3_mem = memory::allocate(engine, P3_lay);
    auto P4_mem = memory::allocate(engine, P4_lay);
    auto P5_mem = memory::allocate(engine, P5_lay);

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
    auto out_ptr = out_mem.pointer<Type>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], static_cast<float>(out_ptr[i])) << "at i = " << i;
    }
}
