// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/experimental_detectron_topk_rois.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <string>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

template <format::type layoutFormat, typename DataType>
struct experimental_detectron_topk_rois_input_types {
    static const auto format = layoutFormat;
    using type = DataType;
};

template <typename EdTopkRoisInput>
struct experimental_detectron_topk_rois_gpu_test : public testing::Test {
    static const auto format = EdTopkRoisInput::format;
    using input_type = typename EdTopkRoisInput::type;
    const ov::element::Type data_type = ov::element::from<input_type>();

    std::vector<input_type> getTypedVector(const std::vector<float>& input) {
        return std::vector<input_type>(input.begin(), input.end());
    }

    void checkOutput(std::shared_ptr<memory> mem, const std::vector<float>& expected_output) {
        cldnn::mem_lock<input_type> out_ptr(mem, get_test_stream());
        ASSERT_EQ(expected_output.size(), out_ptr.size());
        for (size_t i = 0; i < expected_output.size(); ++i) {
            ASSERT_NEAR(static_cast<input_type>(expected_output[i]), out_ptr[i], 0.0001) << "at i = " << i;
        }
    }
};

using format_types = testing::Types<experimental_detectron_topk_rois_input_types<format::bfyx, float>,
                                    experimental_detectron_topk_rois_input_types<format::b_fs_yx_fsv16, float>,
                                    experimental_detectron_topk_rois_input_types<format::b_fs_yx_fsv32, float>,
                                    experimental_detectron_topk_rois_input_types<format::bs_fs_yx_bsv16_fsv16, float>,
                                    experimental_detectron_topk_rois_input_types<format::bs_fs_yx_bsv32_fsv16, float>,
                                    experimental_detectron_topk_rois_input_types<format::bs_fs_yx_bsv32_fsv32, float>,
                                    experimental_detectron_topk_rois_input_types<format::bfyx, ov::float16>,
                                    experimental_detectron_topk_rois_input_types<format::b_fs_yx_fsv16, ov::float16>,
                                    experimental_detectron_topk_rois_input_types<format::b_fs_yx_fsv32, ov::float16>,
                                    experimental_detectron_topk_rois_input_types<format::bs_fs_yx_bsv16_fsv16, ov::float16>,
                                    experimental_detectron_topk_rois_input_types<format::bs_fs_yx_bsv32_fsv16, ov::float16>,
                                    experimental_detectron_topk_rois_input_types<format::bs_fs_yx_bsv32_fsv32, ov::float16>>;

TYPED_TEST_SUITE(experimental_detectron_topk_rois_gpu_test, format_types);

TYPED_TEST(experimental_detectron_topk_rois_gpu_test, check_set_indices_layer) {
    auto& engine = get_test_engine();

    const int rois_num = 2;

    auto roi_input = engine.allocate_memory({this->data_type, format::bfyx, tensor(batch(4), feature(4))});
    auto roi_indices = engine.allocate_memory({data_types::i32, format::bfyx, tensor(batch(rois_num), feature(1))});

    std::vector<float>
        rois{1.0f, 1.0f, 4.0f, 5.0f, 3.0f, 2.0f, 7.0f, 9.0f, 10.0f, 15.0f, 13.0f, 17.0f, 13.0f, 10.0f, 18.0f, 15.0f};
    set_values(roi_input, this->getTypedVector(rois));
    set_values(roi_indices, {3, 1});

    const std::string input_rois_id = "InputRois";
    const std::string input_indices_id = "InputIndices";
    const std::string experimental_detectron_topk_rois_id = "experimental_detectron_topk_rois";
    topology topology;
    topology.add(input_layout(input_rois_id, roi_input->get_layout()));
    topology.add(input_layout(input_indices_id, roi_indices->get_layout()));
    topology.add(reorder("reordered_input", input_info(input_rois_id), this->format, this->data_type));
    topology.add(reorder("reordered_indices", input_info(input_indices_id), this->format, data_types::i32));
    topology.add(experimental_detectron_topk_rois(experimental_detectron_topk_rois_id,
                                                  { input_info("reordered_input"), input_info("reordered_indices") },
                                                  rois_num));
    topology.add(reorder("plane_output", experimental_detectron_topk_rois_id, format::bfyx, this->data_type));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data(input_rois_id, roi_input);
    network.set_input_data(input_indices_id, roi_indices);

    auto result = network.execute();

    std::vector<float> expected_output = {13.0f, 10.0f, 18.0f, 15.0f, 3.0f, 2.0f, 7.0f, 9.0f};

    auto out_mem = result.at("plane_output").get_memory();
    this->checkOutput(out_mem, expected_output);
}

TYPED_TEST(experimental_detectron_topk_rois_gpu_test, check_set_indices_layer_model_less_than_k) {
    auto& engine = get_test_engine();
    // topk is more than model size
    const int rois_num = 3;

    auto roi_input = engine.allocate_memory({this->data_type, format::bfyx, tensor(batch(2), feature(4))});
    auto roi_indices = engine.allocate_memory({data_types::i32, format::bfyx, tensor(batch(2), feature(1))});

    std::vector<float> rois{1.0f, 1.0f, 4.0f, 5.0f, 3.0f, 2.0f, 7.0f, 9.0f};
    set_values(roi_input, this->getTypedVector(rois));
    set_values(roi_indices, {1, 0});

    const std::string input_rois_id = "InputRois";
    const std::string input_indices_id = "InputIndices";
    const std::string experimental_detectron_topk_rois_id = "experimental_detectron_topk_rois";
    topology topology;
    topology.add(input_layout(input_rois_id, roi_input->get_layout()));
    topology.add(input_layout(input_indices_id, roi_indices->get_layout()));
    topology.add(reorder("reordered_input", input_info(input_rois_id), this->format, this->data_type));
    topology.add(reorder("reordered_indices", input_info(input_indices_id), this->format, data_types::i32));
    topology.add(experimental_detectron_topk_rois(experimental_detectron_topk_rois_id,
                                                  { input_info("reordered_input"), input_info("reordered_indices") },
                                                  rois_num));
    topology.add(reorder("plane_output", input_info(experimental_detectron_topk_rois_id), format::bfyx, this->data_type));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data(input_rois_id, roi_input);
    network.set_input_data(input_indices_id, roi_indices);

    auto result = network.execute();

    std::vector<float> expected_output{3.0f, 2.0f, 7.0f, 9.0f, 1.0f, 1.0f, 4.0f, 5.0f};

    auto out_mem = result.at("plane_output").get_memory();
    this->checkOutput(out_mem, expected_output);
}

TEST(experimental_detectron_topk_rois_gpu_test, export_import) {
    const auto test_format = format::bs_fs_yx_bsv32_fsv16;
    const data_types test_data_type = ov::element::from<float>();

    auto& engine = get_test_engine();
    // topk is more than model size
    const int rois_num = 3;

    auto roi_input = engine.allocate_memory({test_data_type, format::bfyx, tensor(batch(2), feature(4))});
    auto roi_indices = engine.allocate_memory({data_types::i32, format::bfyx, tensor(batch(2), feature(1))});

    std::vector<float> rois{1.0f, 1.0f, 4.0f, 5.0f, 3.0f, 2.0f, 7.0f, 9.0f};
    set_values(roi_input, std::vector<float>(rois.begin(), rois.end()));
    set_values(roi_indices, {1, 0});

    const std::string input_rois_id = "InputRois";
    const std::string input_indices_id = "InputIndices";
    const std::string experimental_detectron_topk_rois_id = "experimental_detectron_topk_rois";
    topology topology;
    topology.add(input_layout(input_rois_id, roi_input->get_layout()));
    topology.add(input_layout(input_indices_id, roi_indices->get_layout()));
    topology.add(reorder("reordered_input", input_info(input_rois_id), test_format, test_data_type));
    topology.add(reorder("reordered_indices", input_info(input_indices_id), test_format, data_types::i32));
    topology.add(experimental_detectron_topk_rois(experimental_detectron_topk_rois_id,
                                                  { input_info("reordered_input"), input_info("reordered_indices") },
                                                  rois_num));
    topology.add(reorder("plane_output", input_info(experimental_detectron_topk_rois_id), format::bfyx, test_data_type));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), true);

    network->set_input_data(input_rois_id, roi_input);
    network->set_input_data(input_indices_id, roi_indices);

    auto result = network->execute();

    std::vector<float> expected_output{3.0f, 2.0f, 7.0f, 9.0f, 1.0f, 1.0f, 4.0f, 5.0f};

    auto out_mem = result.at("plane_output").get_memory();
    cldnn::mem_lock<float> out_ptr(out_mem, get_test_stream());
    ASSERT_EQ(expected_output.size(), out_ptr.size());
    for (size_t i = 0; i < expected_output.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(expected_output[i]), out_ptr[i], 0.0001) << "at i = " << i;
    }
}
