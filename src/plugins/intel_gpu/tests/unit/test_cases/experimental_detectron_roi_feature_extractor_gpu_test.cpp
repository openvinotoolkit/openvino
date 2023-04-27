// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp>

#include <cstddef>
#include <iostream>
#include <string>

using namespace cldnn;
using namespace ::tests;

template <typename T>
void test_experimental_detectron_roi_feature_extractor_gpu_fp32_one_level(bool is_caching_test) {
    auto& engine = get_test_engine();

    const int rois_num = 2;
    const int rois_feature_dim = 4;
    const int output_dim = 3;
    const std::vector<int64_t> pyramid_scales = {4};
    const int sampling_ratio = 2;
    const bool aligned = false;
    auto roi_input = engine.allocate_memory({data_types::f32, format::bfyx, tensor(batch(rois_num), feature(rois_feature_dim))});
    auto level_1 = engine.allocate_memory({data_types::f32, format::bfyx, {1, 2, 3, 2}});
    auto second_output = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(batch(rois_num), feature(rois_feature_dim))});

    std::vector<T> rois {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    set_values(roi_input, rois);
    set_values(level_1, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

    const std::string input_rois_id = "InputRois";
    const std::string input_level_1_id = "InputLevel1";
    const std::string second_output_w_id = "second_output_w";
    const std::string second_output_r_id = "second_output_r";
    const std::string feature_extractor_id = "experimental_detectron_roi_feature_extractor";
    const std::string activation_abs_id = "activation_abs";
    topology topology;
    topology.add(input_layout(input_rois_id, roi_input->get_layout()));
    topology.add(input_layout(input_level_1_id, level_1->get_layout()));
    topology.add(mutable_data(second_output_w_id, second_output));
    topology.add(experimental_detectron_roi_feature_extractor(feature_extractor_id,
                                                              { input_info(input_rois_id), input_info(input_level_1_id), input_info(second_output_w_id) },
                                                              output_dim,
                                                              pyramid_scales,
                                                              sampling_ratio,
                                                              aligned));
    topology.add(activation(activation_abs_id, feature_extractor_id,  activation_func::abs));
    topology.add(mutable_data(second_output_r_id, {feature_extractor_id}, second_output));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data(input_rois_id, roi_input);
    network->set_input_data(input_level_1_id, level_1);

    auto outputs = network->execute();

    std::vector<T> expected_first_output {1.416667f, 1.75f, 2.083333f, 2.416667f, 2.75f, 3.083333f, 3.166667f, 3.5f,  3.833333f,
                                              7.416667f, 7.75f, 8.083333f, 8.416667f, 8.75f, 9.083334f, 9.166666f, 9.5f,  9.833334f,
                                              4.166667f, 4.5f,  4.833333f, 4.166667f, 4.5f,  4.833333f, 2.083333f, 2.25f, 2.416667f,
                                              10.16667f, 10.5f, 10.83333f, 10.16667f, 10.5f, 10.83333f, 5.083333f, 5.25f, 5.416667f};

    auto first_network_output = outputs.at(activation_abs_id).get_memory();
    cldnn::mem_lock<T> first_output_ptr(first_network_output, get_test_stream());

    ASSERT_EQ(expected_first_output.size(), first_output_ptr.size());
    for (std::size_t i = 0; i < expected_first_output.size(); i++) {
        ASSERT_FLOAT_EQ(expected_first_output[i], first_output_ptr[i]);
    }

    if (is_caching_test)
        return;

    std::vector<T>& expected_second_output = rois;

    auto second_network_output = outputs.at(second_output_r_id).get_memory();
    ASSERT_TRUE(engine.is_the_same_buffer(*second_output, *second_network_output));
    cldnn::mem_lock<T> second_output_ptr(second_network_output, get_test_stream());

    ASSERT_EQ(expected_second_output.size(), second_output_ptr.size());
    for (std::size_t i = 0; i < expected_second_output.size(); i++) {
        ASSERT_FLOAT_EQ(expected_second_output[i], second_output_ptr[i]);
    }
}

TEST(experimental_detectron_roi_feature_extractor_gpu_fp32, one_level) {
    test_experimental_detectron_roi_feature_extractor_gpu_fp32_one_level<float>(false);
}

TEST(export_import_experimental_detectron_roi_feature_extractor_gpu_fp32, one_level) {
    test_experimental_detectron_roi_feature_extractor_gpu_fp32_one_level<float>(true);
}

TEST(experimental_detectron_roi_feature_extractor_gpu_fp32, two_levels) {
    auto& engine = get_test_engine();

    const int rois_num = 2;
    const int rois_feature_dim = 4;
    const int output_dim = 3;
    const std::vector<int64_t> pyramid_scales = {4, 224};
    const int sampling_ratio = 2;
    const bool aligned = false;
    auto level_1_layout = layout{data_types::f32, format::bfyx, {1, 2, 3, 2}};
    auto level_2_layout = layout{data_types::f32, format::bfyx, {1, 2, 3, 2}};
    auto roi_input = engine.allocate_memory({data_types::f32, format::bfyx, tensor(batch(rois_num), feature(rois_feature_dim))});
    auto level_1 = engine.allocate_memory(level_1_layout);
    auto level_2 = engine.allocate_memory(level_2_layout);
    auto second_output = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(batch(rois_num), feature(rois_feature_dim))});

    auto level_1_padded_layout = level_1_layout;
    level_1_padded_layout.data_padding = padding({0, 0, 1, 1}, {0, 0, 1, 1});
    auto level_2_padded_layout = level_2_layout;
    level_2_padded_layout.data_padding = padding({0, 1, 1, 1}, {0, 1, 1, 1});

    std::vector<float> rois {0.0f, 56.0f, 112.0f, 168.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    set_values(roi_input, rois);
    set_values(level_1, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});
    set_values(level_2, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    const std::string input_rois_id = "InputRois";
    const std::string input_level_1_id = "InputLevel1";
    const std::string input_level_2_id = "InputLevel2";
    const std::string input_level_1_pad_id = "InputLevel1_padding";
    const std::string input_level_2_pad_id = "InputLevel2_padding";
    const std::string second_output_w_id = "second_output_w";
    const std::string second_output_r_id = "second_output_r";
    const std::string feature_extractor_id = "experimental_detectron_roi_feature_extractor";
    const std::string activation_abs_id = "activation_abs";
    topology topology;
    topology.add(input_layout(input_rois_id, roi_input->get_layout()));
    topology.add(input_layout(input_level_1_id, level_1->get_layout()));
    topology.add(input_layout(input_level_2_id, level_2->get_layout()));
    topology.add(reorder(input_level_1_pad_id, input_info(input_level_1_id), level_1_padded_layout));
    topology.add(reorder(input_level_2_pad_id, input_info(input_level_2_id), level_2_padded_layout));
    topology.add(mutable_data(second_output_w_id, second_output));
    topology.add(experimental_detectron_roi_feature_extractor(feature_extractor_id,
                                                              { input_info(input_rois_id),
                                                                input_info(input_level_1_pad_id),
                                                                input_info(input_level_2_pad_id),
                                                                input_info(second_output_w_id) },
                                                              output_dim,
                                                              pyramid_scales,
                                                              sampling_ratio,
                                                              aligned));
    topology.add(activation(activation_abs_id, feature_extractor_id, activation_func::abs));
    topology.add(mutable_data(second_output_r_id, {feature_extractor_id}, second_output));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data(input_rois_id, roi_input);
    network.set_input_data(input_level_1_id, level_1);
    network.set_input_data(input_level_2_id, level_2);

    auto outputs = network.execute();

    std::vector<float> expected_first_output {7.41662f,   7.7499523f, 8.0832853f,  8.41662f,   8.74995f,   9.0832853f, 9.16664f,   9.49998f,   9.83331f,
                                              1.4166187f, 1.7499521f, 2.0832853f,  2.4166186f, 2.7499518f, 3.0832853f, 3.1666427f, 3.4999762f, 3.83331f,
                                              4.166667f,  4.5f,       4.833333f,   4.166667f,  4.5f,       4.833333f,  2.083333f,  2.25f,      2.416667f,
                                              10.16667f,  10.5f,      10.83333f,   10.16667f,  10.5f,      10.83333f,  5.083333f,  5.25f,      5.416667f};

    auto first_network_output = outputs.at(activation_abs_id).get_memory();
    cldnn::mem_lock<float> first_output_ptr(first_network_output, get_test_stream());

    ASSERT_EQ(expected_first_output.size(), first_output_ptr.size());
    for (std::size_t i = 0; i < expected_first_output.size(); i++) {
        ASSERT_FLOAT_EQ(expected_first_output[i], first_output_ptr[i]);
    }

    std::vector<float>& expected_second_output = rois;

    auto second_network_output = outputs.at(second_output_r_id).get_memory();
    ASSERT_TRUE(engine.is_the_same_buffer(*second_output, *second_network_output));
    cldnn::mem_lock<float> second_output_ptr(second_network_output, get_test_stream());

    ASSERT_EQ(expected_second_output.size(), second_output_ptr.size());
    for (std::size_t i = 0; i < expected_second_output.size(); i++) {
        ASSERT_FLOAT_EQ(expected_second_output[i], second_output_ptr[i]);
    }
}

TEST(experimental_detectron_roi_feature_extractor_gpu_fp32, multiple_feature_extractor_op_with_different_number_of_inputs) {
    auto& engine = get_test_engine();

    const int rois_num = 2;
    const int rois_feature_dim = 4;
    const int output_dim = 3;
    const std::vector<int64_t> pyramid_scales_first_instance = {4};
    const std::vector<int64_t> pyramid_scales_second_instance = {4, 224};
    const int sampling_ratio = 2;
    const bool aligned = false;
    auto roi_input_first_instance = engine.allocate_memory({data_types::f32, format::bfyx, tensor(batch(rois_num), feature(rois_feature_dim))});
    auto roi_input_second_instance = engine.allocate_memory({data_types::f32, format::bfyx, tensor(batch(rois_num), feature(rois_feature_dim))});
    auto level_1 = engine.allocate_memory({data_types::f32, format::bfyx, {1, 2, 3, 2}});
    auto level_2 = engine.allocate_memory({data_types::f32, format::bfyx, {1, 2, 3, 2}});
    auto second_output_first_instance = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(batch(rois_num), feature(rois_feature_dim))});
    auto second_output_second_instance = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(batch(rois_num), feature(rois_feature_dim))});

    std::vector<float> rois_first_instance {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::vector<float> rois_second_instance {0.0f, 56.0f, 112.0f, 168.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    set_values(roi_input_first_instance, rois_first_instance);
    set_values(roi_input_second_instance, rois_second_instance);
    set_values(level_1, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});
    set_values(level_2, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    const std::string input_rois_first_instance_id = "InputRois1";
    const std::string input_level_1_first_instance_id = "InputLevel_first_instance";
    const std::string second_output_w_first_instance_id = "second_output_w_first_instance";
    const std::string second_output_r_first_instance_id = "second_output_r_first_instance";
    const std::string feature_extractor_first_instance_id = "experimental_detectron_roi_feature_extractor_1";
    const std::string activation_abs_first_instance_id = "activation_abs_first_instance";
    topology topology;
    topology.add(input_layout(input_rois_first_instance_id, roi_input_first_instance->get_layout()));
    topology.add(input_layout(input_level_1_first_instance_id, level_1->get_layout()));
    topology.add(mutable_data(second_output_w_first_instance_id, second_output_first_instance));
    topology.add(experimental_detectron_roi_feature_extractor(feature_extractor_first_instance_id,
                                                              { input_info(input_rois_first_instance_id), input_info(input_level_1_first_instance_id), input_info(second_output_w_first_instance_id) },
                                                              output_dim,
                                                              pyramid_scales_first_instance,
                                                              sampling_ratio,
                                                              aligned));
    topology.add(activation(activation_abs_first_instance_id, feature_extractor_first_instance_id,  activation_func::abs));
    topology.add(mutable_data(second_output_r_first_instance_id, {feature_extractor_first_instance_id}, second_output_first_instance));

    const std::string input_rois_second_instance_id = "InputRois2";
    const std::string input_level_1_second_instance_id = "InputLevel1_second_instance";
    const std::string input_level_2_second_instance_id = "InputLevel2_second_instance";
    const std::string second_output_w_second_instance_id = "second_output_w_second_instance";
    const std::string second_output_r_second_instance_id = "second_output_r_second_instance";
    const std::string feature_extractor_second_instance_id = "experimental_detectron_roi_feature_extractor_2";
    const std::string activation_abs_second_instance_id = "activation_abs_second_instance";
    topology.add(input_layout(input_rois_second_instance_id, roi_input_second_instance->get_layout()));
    topology.add(input_layout(input_level_1_second_instance_id, level_1->get_layout()));
    topology.add(input_layout(input_level_2_second_instance_id, level_2->get_layout()));
    topology.add(mutable_data(second_output_w_second_instance_id, second_output_second_instance));
    topology.add(experimental_detectron_roi_feature_extractor(feature_extractor_second_instance_id,
                                                              { input_info(input_rois_second_instance_id), input_info(input_level_1_second_instance_id), input_info(input_level_2_second_instance_id), input_info(second_output_w_second_instance_id) },
                                                              output_dim,
                                                              pyramid_scales_second_instance,
                                                              sampling_ratio,
                                                              aligned));
    topology.add(activation(activation_abs_second_instance_id, input_info(feature_extractor_second_instance_id),  activation_func::abs));
    topology.add(mutable_data(second_output_r_second_instance_id, { input_info(feature_extractor_second_instance_id) }, second_output_second_instance));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data(input_rois_first_instance_id, roi_input_first_instance);
    network.set_input_data(input_rois_second_instance_id, roi_input_second_instance);
    network.set_input_data(input_level_1_first_instance_id, level_1);
    network.set_input_data(input_level_1_second_instance_id, level_1);
    network.set_input_data(input_level_2_second_instance_id, level_2);

    auto outputs = network.execute();

    std::vector<float> expected_first_output_first_instance {1.416667f, 1.75f, 2.083333f, 2.416667f, 2.75f, 3.083333f, 3.166667f, 3.5f,  3.833333f,
                                                             7.416667f, 7.75f, 8.083333f, 8.416667f, 8.75f, 9.083334f, 9.166666f, 9.5f,  9.833334f,
                                                             4.166667f, 4.5f,  4.833333f, 4.166667f, 4.5f,  4.833333f, 2.083333f, 2.25f, 2.416667f,
                                                             10.16667f, 10.5f, 10.83333f, 10.16667f, 10.5f, 10.83333f, 5.083333f, 5.25f, 5.416667f};

    auto first_network_output_first_instance = outputs.at(activation_abs_first_instance_id).get_memory();
    cldnn::mem_lock<float> first_output_ptr_first_instance(first_network_output_first_instance, get_test_stream());

    ASSERT_EQ(expected_first_output_first_instance.size(), first_output_ptr_first_instance.size());
    for (std::size_t i = 0; i < expected_first_output_first_instance.size(); i++) {
        ASSERT_FLOAT_EQ(expected_first_output_first_instance[i], first_output_ptr_first_instance[i]);
    }

    std::vector<float>& expected_second_output_first_instance = rois_first_instance;

    auto second_network_output_first_instance = outputs.at(second_output_r_first_instance_id).get_memory();
    ASSERT_TRUE(engine.is_the_same_buffer(*second_output_first_instance, *second_network_output_first_instance));
    cldnn::mem_lock<float> second_output_ptr_first_instance(second_network_output_first_instance, get_test_stream());

    ASSERT_EQ(expected_second_output_first_instance.size(), second_output_ptr_first_instance.size());
    for (std::size_t i = 0; i < expected_second_output_first_instance.size(); i++) {
        ASSERT_FLOAT_EQ(expected_second_output_first_instance[i], second_output_ptr_first_instance[i]);
    }

    std::vector<float> expected_first_output_second_instance {7.41662f,   7.7499523f, 8.0832853f,  8.41662f,   8.74995f,   9.0832853f, 9.16664f,   9.49998f,   9.83331f,
                                                              1.4166187f, 1.7499521f, 2.0832853f,  2.4166186f, 2.7499518f, 3.0832853f, 3.1666427f, 3.4999762f, 3.83331f,
                                                              4.166667f,  4.5f,       4.833333f,   4.166667f,  4.5f,       4.833333f,  2.083333f,  2.25f,      2.416667f,
                                                              10.16667f,  10.5f,      10.83333f,   10.16667f,  10.5f,      10.83333f,  5.083333f,  5.25f,      5.416667f};

    auto first_network_output_second_instance = outputs.at(activation_abs_second_instance_id).get_memory();
    cldnn::mem_lock<float> first_output_ptr_second_instance(first_network_output_second_instance, get_test_stream());

    ASSERT_EQ(expected_first_output_second_instance.size(), first_output_ptr_second_instance.size());
    for (std::size_t i = 0; i < expected_first_output_second_instance.size(); i++) {
        ASSERT_FLOAT_EQ(expected_first_output_second_instance[i], first_output_ptr_second_instance[i]);
    }

    std::vector<float>& expected_second_output_second_instance = rois_second_instance;

    auto second_network_output_second_instance = outputs.at(second_output_r_second_instance_id).get_memory();
    ASSERT_TRUE(engine.is_the_same_buffer(*second_output_second_instance, *second_network_output_second_instance));
    cldnn::mem_lock<float> second_output_ptr_second_instance(second_network_output_second_instance, get_test_stream());

    ASSERT_EQ(expected_second_output_second_instance.size(), second_output_ptr_second_instance.size());
    for (std::size_t i = 0; i < expected_second_output_second_instance.size(); i++) {
        ASSERT_FLOAT_EQ(expected_second_output_second_instance[i], second_output_ptr_second_instance[i]);
    }
}
