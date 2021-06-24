// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/detection_output.hpp>

namespace cldnn {
    template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

using namespace cldnn;
using namespace ::tests;

template <typename T>
class detection_output_test : public ::testing::Test {

public:
    detection_output_test() :
        nms_threshold(0.1f) {}

    void init_buffers(cldnn::memory::ptr prior_memory, cldnn::memory::ptr confidence_memory, cldnn::memory::ptr location_memory,
                      bool share_location, bool variance_encoded_in_target = false,
                      int prior_info_size = 4, int prior_coordinates_offset = 0, bool prior_is_normalized = true) {
        cldnn::mem_lock<T> location_ptr(location_memory, get_test_stream());
        cldnn::mem_lock<T> confidence_ptr(confidence_memory, get_test_stream());
        cldnn::mem_lock<T> prior_box_ptr(prior_memory, get_test_stream());

        T* prior_data = prior_box_ptr.data();
        T* confidence_data = confidence_ptr.data();
        T* location_data = location_ptr.data();

        // Fill prior-box data.
        const float step = 0.5f;
        const float box_size = 0.3f;
        const float prior_multiplier = prior_is_normalized ? 1.0f : static_cast<float>(this->img_size);
        const float variance = 0.1f;
        int idx = 0;
        for (int h = 0; h < 2; ++h) {
            float center_y = (h + 0.5f) * step;
            for (int w = 0; w < 2; ++w) {
                float center_x = (w + 0.5f) * step;
                prior_data[idx+prior_coordinates_offset+0] = (center_x - box_size / 2) * prior_multiplier;
                prior_data[idx+prior_coordinates_offset+1] = (center_y - box_size / 2) * prior_multiplier;
                prior_data[idx+prior_coordinates_offset+2] = (center_x + box_size / 2) * prior_multiplier;
                prior_data[idx+prior_coordinates_offset+3] = (center_y + box_size / 2) * prior_multiplier;

                idx += prior_info_size;
            }
        }
        if (!variance_encoded_in_target) {
            for (int i = 0; i < idx; ++i) {
                prior_data[idx + i] = variance;
            }
        }

        // Fill confidences.
        idx = 0;
        for (int i = 0; i < num_of_images; ++i) {
            for (int j = 0; j < num_priors; ++j) {
                for (int c = 0; c < num_classes; ++c) {
                    if (i % 2 == c % 2) {
                        confidence_data[idx++] = j * 0.2f;
                    } else {
                        confidence_data[idx++] = 1 - j * 0.2f;
                    }
                }
            }
        }

        // Fill locations.
        const int num_loc_classes = share_location ? 1 : num_classes;
        const float loc_multiplier = variance_encoded_in_target ? variance : 1.0f;
        idx = 0;
        for (int i = 0; i < num_of_images; ++i) {
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    for (int c = 0; c < num_loc_classes; ++c) {
                        location_data[idx++] = (w % 2 ? -1 : 1) * (i * 1 + c / 2.f + 0.5f) * loc_multiplier;
                        location_data[idx++] = (h % 2 ? -1 : 1) * (i * 1 + c / 2.f + 0.5f) * loc_multiplier;
                        location_data[idx++] = (w % 2 ? -1 : 1) * (i * 1 + c / 2.f + 0.5f) * loc_multiplier;
                        location_data[idx++] = (h % 2 ? -1 : 1) * (i * 1 + c / 2.f + 0.5f) * loc_multiplier;
                    }
                }
            }
        }
    }

    void init_buffer_sort(cldnn::memory::ptr input_buff) {
        cldnn::mem_lock<T> input_data_ptr(input_buff, get_test_stream());

        EXPECT_EQ((int)input_buff->count(), 128);

        T* input_data = input_data_ptr.data();
        input_data[0] = 8;
        input_data[1] = 3;
        input_data[16] = 0; input_data[17] = 0; input_data[18] = 0.6f; input_data[19] = 0.55f; input_data[20] = 0.55f; input_data[21] = 0.85f; input_data[22] = 0.85f;
        input_data[23] = 0; input_data[24] = 0; input_data[25] = 0.4f; input_data[26] = 0.15f; input_data[27] = 0.55f; input_data[28] = 0.45f; input_data[29] = 0.85f;
        input_data[30] = 0; input_data[31] = 0; input_data[32] = 0.2f; input_data[33] = 0.55f; input_data[34] = 0.15f; input_data[35] = 0.85f; input_data[36] = 0.45f;
        input_data[37] = 0; input_data[38] = 0; input_data[39] = 0.0f; input_data[40] = 0.15f; input_data[41] = 0.15f; input_data[42] = 0.45f; input_data[43] = 0.45f;
        input_data[44] = 0; input_data[45] = 1; input_data[46] = 1.0f; input_data[47] = 0.20f; input_data[48] = 0.20f; input_data[49] = 0.50f; input_data[50] = 0.50f;
        input_data[51] = 0; input_data[52] = 1; input_data[53] = 0.8f; input_data[54] = 0.50f; input_data[55] = 0.20f; input_data[56] = 0.80f; input_data[57] = 0.50f;
        input_data[58] = 0; input_data[59] = 1; input_data[60] = 0.6f; input_data[61] = 0.20f; input_data[62] = 0.50f; input_data[63] = 0.50f; input_data[64] = 0.80f;
        input_data[65] = 0; input_data[66] = 1; input_data[67] = 0.4f; input_data[68] = 0.50f; input_data[69] = 0.50f; input_data[70] = 0.80f; input_data[71] = 0.80f;
        input_data[72] = 1; input_data[73] = 0; input_data[74] = 1.0f; input_data[75] = 0.25f; input_data[76] = 0.25f; input_data[77] = 0.55f; input_data[78] = 0.55f;
        input_data[79] = 1; input_data[80] = 0; input_data[81] = 0.4f; input_data[82] = 0.45f; input_data[83] = 0.45f; input_data[84] = 0.75f; input_data[85] = 0.75f;
        input_data[86] = -1; input_data[87] = 0; input_data[88] = 0; input_data[89] = 0; input_data[90] = 0; input_data[91] = 0; input_data[92] = 0;
        input_data[93] = -1; input_data[94] = 0; input_data[95] = 0; input_data[96] = 0; input_data[97] = 0; input_data[98] = 0; input_data[99] = 0;
        input_data[100] = 1; input_data[101] = 1; input_data[102] = 0.6f; input_data[103] = 0.40f; input_data[104] = 0.40f; input_data[105] = 0.70f; input_data[106] = 0.70f;
        input_data[107] = -1; input_data[108] = 0; input_data[109] = 0; input_data[110] = 0; input_data[111] = 0; input_data[112] = 0; input_data[113] = 0;
        input_data[114] = -1; input_data[115] = 0; input_data[116] = 0; input_data[117] = 0; input_data[118] = 0; input_data[119] = 0; input_data[120] = 0;
        input_data[121] = -1; input_data[122] = 0; input_data[123] = 0; input_data[124] = 0; input_data[125] = 0; input_data[126] = 0; input_data[127] = 0;
    }

    void check_results(const memory::ptr output, const int num, const std::string values) {
        assert(num < output->get_layout().size.spatial[1]);

        // Split values to vector of items.
        std::vector<std::string> items;
        std::istringstream iss(values);
        std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(), back_inserter(items));
        EXPECT_EQ((int)items.size(), 7);

        // Check data.
        cldnn::mem_lock<T> out_ptr(output, get_test_stream());
        const T* data = out_ptr.data();
        for (int i = 0; i < 2; ++i) {
            EXPECT_EQ(static_cast<int>((float)data[num * output->get_layout().size.spatial[0] + i]), atoi(items[i].c_str()));
        }
        for (int i = 2; i < 7; ++i) {
            EXPECT_TRUE(floating_point_equal(data[num * output->get_layout().size.spatial[0] + i], (T)(float)atof(items[i].c_str())));
        }
    }

    void setup_basic() {
        const bool share_location = true;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 150;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 2, 1, this->num_priors * 4 } });

        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));

        topology.add(detection_output("detection_output", "input_location", "input_confidence", "input_prior_box", this->num_classes, keep_top_k));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "detection_output");

        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.batch[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.feature[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[0], 7);
    }

    void setup_two_layers() {
        const bool share_location = true;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 150;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 2, 1, this->num_priors * 4 } });

        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));

        topology.add(detection_output("detection_output_1", "input_location", "input_confidence", "input_prior_box", this->num_classes, keep_top_k));
        topology.add(detection_output("detection_output_2", "input_location", "input_confidence", "input_prior_box", this->num_classes, keep_top_k));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(2));
        unsigned i = 1;
        for (auto it = outputs.begin(); it != outputs.begin(); it++) {

            EXPECT_EQ(it->first, "detection_output_" + std::to_string(i));

            EXPECT_EQ(it->second.get_memory()->get_layout().size.batch[0], 1);
            EXPECT_EQ(it->second.get_memory()->get_layout().size.feature[0], 1);
            EXPECT_EQ(it->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
            EXPECT_EQ(it->second.get_memory()->get_layout().size.spatial[0], 7);
            i++;
        }
    }

    void forward_share_location() {
        const bool share_location = true;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 4;
        const int background_label_id = 0;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 2, 1, this->num_priors * 4 } });

        this->init_buffers(input_prior_box, input_confidence, input_location, share_location);

        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));

        topology.add(detection_output("detection_output", "input_location", "input_confidence", "input_prior_box", this->num_classes, keep_top_k, share_location, background_label_id, this->nms_threshold));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "detection_output");

        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.batch[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.feature[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[0], 7);

        auto output_prim = outputs.begin()->second.get_memory();

        check_results(output_prim, 0, "0 1 1.0 0.15 0.15 0.45 0.45");
        check_results(output_prim, 1, "0 1 0.8 0.55 0.15 0.85 0.45");
        check_results(output_prim, 2, "0 1 0.6 0.15 0.55 0.45 0.85");
        check_results(output_prim, 3, "0 1 0.4 0.55 0.55 0.85 0.85");
        check_results(output_prim, 4, "1 1 0.6 0.45 0.45 0.75 0.75");
        check_results(output_prim, 5, "1 1 0.0 0.25 0.25 0.55 0.55");
        check_results(output_prim, 6, "-1 0 0 0 0 0 0");
        check_results(output_prim, 7, "-1 0 0 0 0 0 0");
    }

    void forward_num_detections_greater_than_keep_top_k() {
        const bool share_location = true;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 1;
        const int background_label_id = 0;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 2, 1, this->num_priors * 4 } });

        this->init_buffers(input_prior_box, input_confidence, input_location, share_location);

        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));

        topology.add(detection_output("detection_output", "input_location", "input_confidence", "input_prior_box", this->num_classes, keep_top_k, share_location, background_label_id, this->nms_threshold));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "detection_output");

        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.batch[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.feature[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[0], 7);

        auto output_prim = outputs.begin()->second.get_memory();

        check_results(output_prim, 0, "0 1 1.0 0.15 0.15 0.45 0.45");
        check_results(output_prim, 1, "1 1 0.6 0.45 0.45 0.75 0.75");
    }

    void forward_num_detections_smaller_than_keep_top_k() {
        const bool share_location = true;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 6;
        const int background_label_id = 0;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 2, 1, this->num_priors * 4 } });

        this->init_buffers(input_prior_box, input_confidence, input_location, share_location);

        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));

        topology.add(detection_output("detection_output", "input_location", "input_confidence", "input_prior_box", this->num_classes, keep_top_k, share_location, background_label_id, this->nms_threshold));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "detection_output");

        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.batch[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.feature[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[0], 7);

        auto output_prim = outputs.begin()->second.get_memory();

        check_results(output_prim, 0, "0 1 1.0 0.15 0.15 0.45 0.45");
        check_results(output_prim, 1, "0 1 0.8 0.55 0.15 0.85 0.45");
        check_results(output_prim, 2, "0 1 0.6 0.15 0.55 0.45 0.85");
        check_results(output_prim, 3, "0 1 0.4 0.55 0.55 0.85 0.85");
        check_results(output_prim, 4, "1 1 0.6 0.45 0.45 0.75 0.75");
        check_results(output_prim, 5, "1 1 0.0 0.25 0.25 0.55 0.55");
        check_results(output_prim, 6, "-1 0 0 0 0 0 0");
        check_results(output_prim, 7, "-1 0 0 0 0 0 0");
        check_results(output_prim, 8, "-1 0 0 0 0 0 0");
        check_results(output_prim, 9, "-1 0 0 0 0 0 0");
        check_results(output_prim, 10, "-1 0 0 0 0 0 0");
        check_results(output_prim, 11, "-1 0 0 0 0 0 0");
    }

    void test_forward_share_location_top_k() {
        const bool share_location = true;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 2;
        const int top_k = 2;
        const int background_label_id = 0;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 2, 1, this->num_priors * 4 } });

        this->init_buffers(input_prior_box, input_confidence, input_location, share_location);

        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));

        topology.add(detection_output("detection_output", "input_location", "input_confidence", "input_prior_box", this->num_classes, keep_top_k, share_location, background_label_id, this->nms_threshold, top_k));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "detection_output");

        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.batch[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.feature[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[0], 7);

        auto output_prim = outputs.begin()->second.get_memory();

        check_results(output_prim, 0, "0 1 1.0 0.15 0.15 0.45 0.45");
        check_results(output_prim, 1, "0 1 0.8 0.55 0.15 0.85 0.45");
        check_results(output_prim, 2, "1 1 0.6 0.45 0.45 0.75 0.75");
        check_results(output_prim, 3, "-1 0 0 0 0 0 0");
    }

    void forward_no_share_location() {
        const bool share_location = false;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 10;
        const int background_label_id = -1;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 2, 1, this->num_priors * 4 } });

        this->init_buffers(input_prior_box, input_confidence, input_location, share_location);

        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));

        topology.add(detection_output("detection_output", "input_location", "input_confidence", "input_prior_box", this->num_classes, keep_top_k, share_location, background_label_id, this->nms_threshold));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "detection_output");

        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.batch[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.feature[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[0], 7);

        auto output_prim = outputs.begin()->second.get_memory();

        check_results(output_prim, 0, "0 0 0.6 0.55 0.55 0.85 0.85");
        check_results(output_prim, 1, "0 0 0.4 0.15 0.55 0.45 0.85");
        check_results(output_prim, 2, "0 0 0.2 0.55 0.15 0.85 0.45");
        check_results(output_prim, 3, "0 0 0.0 0.15 0.15 0.45 0.45");
        check_results(output_prim, 4, "0 1 1.0 0.20 0.20 0.50 0.50");
        check_results(output_prim, 5, "0 1 0.8 0.50 0.20 0.80 0.50");
        check_results(output_prim, 6, "0 1 0.6 0.20 0.50 0.50 0.80");
        check_results(output_prim, 7, "0 1 0.4 0.50 0.50 0.80 0.80");
        check_results(output_prim, 8, "1 0 1.0 0.25 0.25 0.55 0.55");
        check_results(output_prim, 9, "1 0 0.4 0.45 0.45 0.75 0.75");
        check_results(output_prim, 10, "1 1 0.6 0.40 0.40 0.70 0.70");
        check_results(output_prim, 11, "-1 0 0 0 0 0 0");
        check_results(output_prim, 12, "-1 0 0 0 0 0 0");
        check_results(output_prim, 13, "-1 0 0 0 0 0 0");
        check_results(output_prim, 14, "-1 0 0 0 0 0 0");
        check_results(output_prim, 15, "-1 0 0 0 0 0 0");
        check_results(output_prim, 16, "-1 0 0 0 0 0 0");
        check_results(output_prim, 17, "-1 0 0 0 0 0 0");
        check_results(output_prim, 18, "-1 0 0 0 0 0 0");
        check_results(output_prim, 19, "-1 0 0 0 0 0 0");
    }

    void forward_no_share_location_top_k() {
        const bool share_location = false;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 4;
        const int background_label_id = -1;
        const int top_k = 2;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 2, 1, this->num_priors * 4 } });

        this->init_buffers(input_prior_box, input_confidence, input_location, share_location);

        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));

        topology.add(detection_output("detection_output", "input_location", "input_confidence", "input_prior_box", this->num_classes, keep_top_k, share_location, background_label_id, this->nms_threshold, top_k));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "detection_output");

        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.batch[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.feature[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[0], 7);

        auto output_prim = outputs.begin()->second.get_memory();

        check_results(output_prim, 0, "0 0 0.6 0.55 0.55 0.85 0.85");
        check_results(output_prim, 1, "0 0 0.4 0.15 0.55 0.45 0.85");
        check_results(output_prim, 2, "0 1 1.0 0.20 0.20 0.50 0.50");
        check_results(output_prim, 3, "0 1 0.8 0.50 0.20 0.80 0.50");
        check_results(output_prim, 4, "1 0 1.0 0.25 0.25 0.55 0.55");
        check_results(output_prim, 5, "1 1 0.6 0.40 0.40 0.70 0.70");
        check_results(output_prim, 6, "-1 0 0 0 0 0 0");
        check_results(output_prim, 7, "-1 0 0 0 0 0 0");
    }

    void forward_no_share_location_neg_0() {
        const bool share_location = false;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 5;
        const int background_label_id = 0;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 2, 1, this->num_priors * 4 } });

        this->init_buffers(input_prior_box, input_confidence, input_location, share_location);

        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));

        topology.add(detection_output("detection_output", "input_location", "input_confidence", "input_prior_box", this->num_classes, keep_top_k, share_location, background_label_id, this->nms_threshold));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "detection_output");

        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.batch[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.feature[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[0], 7);

        auto output_prim = outputs.begin()->second.get_memory();

        check_results(output_prim, 0, "0 1 1.0 0.20 0.20 0.50 0.50");
        check_results(output_prim, 1, "0 1 0.8 0.50 0.20 0.80 0.50");
        check_results(output_prim, 2, "0 1 0.6 0.20 0.50 0.50 0.80");
        check_results(output_prim, 3, "0 1 0.4 0.50 0.50 0.80 0.80");
        check_results(output_prim, 4, "1 1 0.6 0.40 0.40 0.70 0.70");
        check_results(output_prim, 5, "-1 0 0 0 0 0 0");
        check_results(output_prim, 6, "-1 0 0 0 0 0 0");
        check_results(output_prim, 7, "-1 0 0 0 0 0 0");
        check_results(output_prim, 8, "-1 0 0 0 0 0 0");
        check_results(output_prim, 9, "-1 0 0 0 0 0 0");
    }

    void forward_no_share_location_neg_0_top_k() {
        const bool share_location = false;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 2;
        const int background_label_id = 0;
        const int top_k = 2;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 2, 1, this->num_priors * 4 } });

        this->init_buffers(input_prior_box, input_confidence, input_location, share_location);

        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));

        topology.add(detection_output("detection_output", "input_location", "input_confidence", "input_prior_box", this->num_classes, keep_top_k, share_location, background_label_id, this->nms_threshold, top_k));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "detection_output");

        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.batch[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.feature[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[0], 7);

        auto output_prim = outputs.begin()->second.get_memory();

        check_results(output_prim, 0, "0 1 1.0 0.20 0.20 0.50 0.50");
        check_results(output_prim, 1, "0 1 0.8 0.50 0.20 0.80 0.50");
        check_results(output_prim, 2, "1 1 0.6 0.40 0.40 0.70 0.70");
        check_results(output_prim, 3, "-1 0 0 0 0 0 0");
    }

    void forward_no_share_location_top_k_input_padding() {
        const bool share_location = false;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 4;
        const int background_label_id = -1;
        const int top_k = 2;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 2, 1, this->num_priors * 4 } });

        this->init_buffers(input_prior_box, input_confidence, input_location, share_location);
        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));
        topology.add(reorder("input_location_padded", "input_location", input_location->get_layout().with_padding(padding{ { 0, 0, 12, 3 },{ 0, 0, 5, 11 } })));
        topology.add(reorder("input_confidence_padded", "input_confidence", input_location->get_layout().with_padding(padding{ { 0, 0, 2, 7 },{ 0, 0, 13, 1 } })));

        topology.add(detection_output("detection_output", "input_location_padded", "input_confidence_padded", "input_prior_box", this->num_classes, keep_top_k, share_location, background_label_id, this->nms_threshold, top_k));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "detection_output");

        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.batch[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.feature[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[0], 7);

        auto output_prim = outputs.begin()->second.get_memory();

        check_results(output_prim, 0, "0 0 0.6 0.55 0.55 0.85 0.85");
        check_results(output_prim, 1, "0 0 0.4 0.15 0.55 0.45 0.85");
        check_results(output_prim, 2, "0 1 1.0 0.20 0.20 0.50 0.50");
        check_results(output_prim, 3, "0 1 0.8 0.50 0.20 0.80 0.50");
        check_results(output_prim, 4, "1 0 1.0 0.25 0.25 0.55 0.55");
        check_results(output_prim, 5, "1 1 0.6 0.40 0.40 0.70 0.70");
        check_results(output_prim, 6, "-1 0 0 0 0 0 0");
        check_results(output_prim, 7, "-1 0 0 0 0 0 0");
    }

    void test_forward_no_share_location_top_k_faster_rcnn_case() {
        const bool share_location = false;
        const int num_loc_classes = share_location ? 1 : this->num_classes;
        const int keep_top_k = 4;
        const int background_label_id = -1;
        const int top_k = 2;
        const float eta = 1.0f;
        const prior_box_code_type code_type = prior_box_code_type::corner;
        const bool variance_encoded_in_target = true;
        const float confidence_threshold = -std::numeric_limits<float>::max();
        const int32_t prior_info_size = 5;
        const int32_t prior_coordinates_offset = 1;
        const bool prior_is_normalized = true;

        auto& engine = get_test_engine();
        cldnn::memory::ptr input_location = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * num_loc_classes * 4, 1, 1 } });
        cldnn::memory::ptr input_confidence = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ this->num_of_images, this->num_priors * this->num_classes, 1, 1 } });
        cldnn::memory::ptr input_prior_box = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx,{ 1, 1, 1, this->num_priors * prior_info_size } });

        this->init_buffers(input_prior_box, input_confidence, input_location, share_location, variance_encoded_in_target,
            prior_info_size, prior_coordinates_offset, prior_is_normalized);

        topology topology;
        topology.add(input_layout("input_location", input_location->get_layout()));
        topology.add(input_layout("input_confidence", input_confidence->get_layout()));
        topology.add(input_layout("input_prior_box", input_prior_box->get_layout()));
        topology.add(reorder("input_location_padded", "input_location", input_location->get_layout().with_padding(padding{ { 0, 0, 12, 3 },{ 0, 0, 5, 11 } })));
        topology.add(reorder("input_confidence_padded", "input_confidence", input_location->get_layout().with_padding(padding{ { 0, 0, 2, 7 },{ 0, 0, 13, 1 } })));

        topology.add(detection_output("detection_output", "input_location_padded", "input_confidence_padded", "input_prior_box",
            this->num_classes, keep_top_k, share_location, background_label_id, this->nms_threshold, top_k,
            eta, code_type, variance_encoded_in_target, confidence_threshold, prior_info_size, prior_coordinates_offset,
            prior_is_normalized, this->img_size, this->img_size
        ));

        build_options opts;
        network network(engine, topology, opts);
        network.set_input_data("input_location", input_location);
        network.set_input_data("input_confidence", input_confidence);
        network.set_input_data("input_prior_box", input_prior_box);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "detection_output");

        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.batch[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.feature[0], 1);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[1], keep_top_k * this->num_of_images);
        EXPECT_EQ(outputs.begin()->second.get_memory()->get_layout().size.spatial[0], 7);

        auto output_prim = outputs.begin()->second.get_memory();

        check_results(output_prim, 0, "0 0 0.6 0.55 0.55 0.85 0.85");
        check_results(output_prim, 1, "0 0 0.4 0.15 0.55 0.45 0.85");
        check_results(output_prim, 2, "0 1 1.0 0.20 0.20 0.50 0.50");
        check_results(output_prim, 3, "0 1 0.8 0.50 0.20 0.80 0.50");
        check_results(output_prim, 4, "1 0 1.0 0.25 0.25 0.55 0.55");
        check_results(output_prim, 5, "1 1 0.6 0.40 0.40 0.70 0.70");
        check_results(output_prim, 6, "-1 0 0 0 0 0 0");
        check_results(output_prim, 7, "-1 0 0 0 0 0 0");
    }

    static const int num_of_images = 2;
    static const int num_classes = 2;
    static const int num_priors = 4;
    static const int img_size = 300;
    const float nms_threshold;
};

typedef ::testing::Types<float, FLOAT16> detection_output_test_types;
TYPED_TEST_SUITE(detection_output_test, detection_output_test_types);

TYPED_TEST(detection_output_test, test_setup_basic) {
    this->setup_basic();
}

TYPED_TEST(detection_output_test, test_setup_two_layers) {
    this->setup_two_layers();
}

TYPED_TEST(detection_output_test, test_forward_share_location) {
    this->forward_share_location();
}

TYPED_TEST(detection_output_test, test_forward_num_detections_greater_than_keep_top_k) {
    this->forward_num_detections_greater_than_keep_top_k();
}

TYPED_TEST(detection_output_test, test_forward_num_detections_smaller_than_keep_top_k) {
    this->forward_num_detections_smaller_than_keep_top_k();
}

TYPED_TEST(detection_output_test, test_forward_share_location_top_k) {
    this->test_forward_share_location_top_k();
}

TYPED_TEST(detection_output_test, test_forward_no_share_location) {
    this->forward_no_share_location();
}

TYPED_TEST(detection_output_test, test_forward_no_share_location_top_k) {
    this->forward_no_share_location_top_k();
}

TYPED_TEST(detection_output_test, test_forward_no_share_location_neg_0) {
    this->forward_no_share_location_neg_0();
}

TYPED_TEST(detection_output_test, test_forward_no_share_location_neg_0_top_k) {
    this->forward_no_share_location_neg_0_top_k();
}

TYPED_TEST(detection_output_test, test_forward_no_share_location_top_k_input_padding) {
    this->forward_no_share_location_top_k_input_padding();
}

TYPED_TEST(detection_output_test, test_forward_no_share_location_top_k_faster_rcnn_case) {
    this->test_forward_no_share_location_top_k_faster_rcnn_case();
}
