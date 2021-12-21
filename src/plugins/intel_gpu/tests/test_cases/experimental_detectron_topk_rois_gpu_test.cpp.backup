// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/experimental_detectron_topk_rois.hpp>

#include <cstddef>
#include <string>

using namespace cldnn;
using namespace ::tests;

TEST(experimental_detectron_topk_rois_gpu_fp32, check_set_indices_layer) {
    auto &engine = get_test_engine();

    const int rois_num = 2;

    auto roi_input = engine.allocate_memory(
            {data_types::f32, format::bfyx, tensor(batch(4), feature(4))});
    auto roi_indices = engine.allocate_memory({data_types::i32, format::bfyx, tensor(batch(rois_num), feature(1))});

    std::vector<float> rois{1.0f, 1.0f, 4.0f, 5.0f,
                            3.0f, 2.0f, 7.0f, 9.0f,
                            10.0f, 15.0f, 13.0f, 17.0f,
                            13.0f, 10.0f, 18.0f, 15.0f};
    set_values(roi_input, rois);
    set_values(roi_indices,
               {3, 1});

    const std::string input_rois_id = "InputRois";
    const std::string input_indices_id = "InputIndices";;
    const std::string experimental_detectron_topk_rois_id = "experimental_detectron_topk_rois";
    topology topology;
    topology.add(input_layout(input_rois_id, roi_input->get_layout()));
    topology.add(input_layout(input_indices_id, roi_indices->get_layout()));

    topology.add(experimental_detectron_topk_rois(experimental_detectron_topk_rois_id,
                                                  {input_rois_id, input_indices_id}, rois_num));

    network network(engine, topology);

    network.set_input_data(input_rois_id, roi_input);
    network.set_input_data(input_indices_id, roi_indices);

    auto result = network.execute();

    std::vector<float> expected_output{13.0f, 10.0f, 18.0f, 15.0f,
                                       3.0f, 2.0f, 7.0f, 9.0f};

    auto out_mem = result.at(experimental_detectron_topk_rois_id).get_memory();
    cldnn::mem_lock<float> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(expected_output.size(), out_ptr.size());
    for (size_t i = 0; i < expected_output.size(); ++i) {
        EXPECT_NEAR(expected_output[i], out_ptr[i], 0.0001) << "at i = " << i;
    }
}

TEST(experimental_detectron_topk_rois_gpu_fp32, check_set_indices_layer_model_less_than_k) {
    auto &engine = get_test_engine();
    // topk is more than model size
    const int rois_num = 3;

    auto roi_input = engine.allocate_memory(
            {data_types::f32, format::bfyx, tensor(batch(2), feature(4))});
    auto roi_indices = engine.allocate_memory({data_types::i32, format::bfyx, tensor(batch(2), feature(1))});

    std::vector<float> rois{1.0f, 1.0f, 4.0f, 5.0f,
                            3.0f, 2.0f, 7.0f, 9.0f};
    set_values(roi_input, rois);
    set_values(roi_indices,
               {1, 0});

    const std::string input_rois_id = "InputRois";
    const std::string input_indices_id = "InputIndices";;
    const std::string experimental_detectron_topk_rois_id = "experimental_detectron_topk_rois";
    topology topology;
    topology.add(input_layout(input_rois_id, roi_input->get_layout()));
    topology.add(input_layout(input_indices_id, roi_indices->get_layout()));

    topology.add(experimental_detectron_topk_rois(experimental_detectron_topk_rois_id,
                                                  {input_rois_id, input_indices_id}, rois_num));

    network network(engine, topology);

    network.set_input_data(input_rois_id, roi_input);
    network.set_input_data(input_indices_id, roi_indices);

    auto result = network.execute();

    std::vector<float> expected_output{3.0f, 2.0f, 7.0f, 9.0f,
                                       1.0f, 1.0f, 4.0f, 5.0f};


    auto out_mem = result.at(experimental_detectron_topk_rois_id).get_memory();
    cldnn::mem_lock<float> out_ptr(out_mem, get_test_stream());

    ASSERT_EQ(expected_output.size(), out_ptr.size());
    for (size_t i = 0; i < expected_output.size(); ++i) {
        EXPECT_NEAR(expected_output[i], out_ptr[i], 0.0001) << "at i = " << i;
    }
}
