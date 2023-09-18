// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/rms.hpp>
#include "rms_inst.h"

using namespace cldnn;
using namespace ::tests;

class rms_gpu_test : public ::testing::TestWithParam<cldnn::format> {};

template <typename T>
void rms_ref(const memory::ptr input, const memory::ptr gamma, memory::ptr output, float epsilon) {
    auto input_layout = input->get_layout();
    auto gamma_layout = gamma->get_layout();

    uint32_t batch_size = input_layout.batch();
    uint32_t feature_size = input_layout.feature();
    uint32_t y_size = input_layout.spatial(1);
    uint32_t x_size = input_layout.spatial(0);

    cldnn::mem_lock<T> src(input, get_test_stream());
    cldnn::mem_lock<T> weight(gamma, get_test_stream());
    cldnn::mem_lock<T> dst(output, get_test_stream());

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t f = 0; f < feature_size; ++f) {
            float rms = 0.f;
            for (uint32_t y = 0; y < y_size; ++y) {
                for (uint32_t x = 0; x < x_size; ++x) {
                    auto tensor_src = tensor(batch(b), feature(f), spatial(x, y, 0, 0));
                    size_t src_offset = input_layout.get_linear_offset(tensor_src);
                    rms += std::pow(static_cast<float>(src[src_offset]), 2);
                }
            }
            rms /= y_size * x_size;
            rms += epsilon;
            rms = std::pow(std::sqrt(rms), -1);

            for (uint32_t y = 0; y < y_size; ++y) {
                for (uint32_t x = 0; x < x_size; ++x) {
                    auto tensor_src = tensor(batch(b), feature(f), spatial(x, y, 0, 0));
                    auto tensor_weight = tensor(batch(b), feature(0), spatial(x, y, 0, 0));
                    auto tensor_dst = tensor(batch(b), feature(f), spatial(x, y, 0, 0));
                    size_t src_offset = input_layout.get_linear_offset(tensor_src);
                    size_t weight_offset = input_layout.get_linear_offset(tensor_weight);
                    size_t dst_offset = input_layout.get_linear_offset(tensor_dst);
                    float result = rms * static_cast<float>(src[src_offset]) * static_cast<float>(weight[weight_offset]);
                    dst[dst_offset] = result;
                }
            }
        }
    }
}

TEST(rms_gpu_test, rms_test_bfyx) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ov::PartialShape{1, 2, 6}, data_types::f32, format::bfyx});
    auto gamma = engine.allocate_memory({ov::PartialShape{1, 6}, data_types::f32, format::bfyx});
    auto output_ref = engine.allocate_memory({ov::PartialShape{1, 2, 6}, data_types::f32, format::bfyx});

    set_values(input, {
        0.029785f, 0.014038f, 0.003098f, 0.013123f, 0.015137f, 0.009399f,
        0.008362f, 0.008179f, 0.018188f, 0.021973f, 0.005249f, 0.004639f
    });
    set_values(gamma, {
        0.029785f, 0.014038f, 0.003098f, 0.013123f, 0.015137f, 0.009399f
    });

    rms_ref<float>(input, gamma, output_ref, 1e-5f);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("gamma", gamma->get_layout()));
    topology.add(rms("rms", input_info("input"), input_info("gamma"), 1e-5f));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("gamma", gamma);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "rms");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        ASSERT_EQ(output_ptr[i], output_ref_ptr[i]);
    }
}
