// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/region_yolo.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/runtime/memory.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

namespace {
inline int entry_index(int width,
                       int height,
                       int coords,
                       int classes,
                       int outputs,
                       int batch,
                       int location,
                       int entry) {
    int n = location / (width * height);
    int loc = location % (width * height);
    return batch * outputs + n * width * height * (coords + classes + 1) + entry * width * height + loc;
}

template <typename T>
inline T sigmoid(float x) {
    return static_cast<T>(1.f / (1.f + std::exp(-x)));
}

template <typename T>
inline void softmax_generic(const T* src_data, T* dst_data, uint32_t batches, uint32_t channels, uint32_t height, uint32_t width) {
    const uint32_t area = height * width;
    for (unsigned int batch_idx = 0; batch_idx < batches; batch_idx++)
    {
        const int offset = batch_idx * channels * area;
        for (unsigned int i = 0; i < height * width; i++)
        {
            T max = src_data[batch_idx * channels * area + i];
            for (unsigned int channel_idx = 0; channel_idx < channels; channel_idx++)
            {
                T val = src_data[offset + channel_idx * area + i];
                max = std::max(max, val);
            }

            T sum = 0;
            for (unsigned int channel_idx = 0; channel_idx < channels; channel_idx++)
            {
                dst_data[offset + channel_idx * area + i] =
                    std::exp((float)(src_data[offset + channel_idx * area + i] - max));
                sum += dst_data[offset + channel_idx * area + i];
            }

            for (unsigned int channel_idx = 0; channel_idx < channels; channel_idx++)
            {
                dst_data[offset + channel_idx * area + i] /= sum;
            }
        }
    }
}

uint32_t shape_size(const std::vector<uint32_t>& input_shape) {
    uint32_t ret = 1;
    std::for_each(input_shape.begin(), input_shape.end(), [&ret](uint32_t n){
        ret *= n;
    });

    return ret;
}

template <typename T>
void region_yolo_ref(const T* input,
                     T* output,
                     const std::vector<uint32_t>& input_shape,
                     const uint32_t coords,
                     const uint32_t classes,
                     const uint32_t regions,
                     const bool do_softmax,
                     const std::vector<int64_t>& mask) {
    ASSERT_EQ(input_shape.size(), 4);

    const uint32_t batches = input_shape[0];
    //const uint32_t channels = input_shape[1];
    const uint32_t height = input_shape[2];
    const uint32_t width = input_shape[3];

    const auto mask_size = mask.size();

    std::copy(input, input + shape_size(input_shape), output);

    uint32_t num_regions = 0;
    uint32_t end_index = 0;

    if (do_softmax) {
        // Region layer (Yolo v2)
        num_regions = regions;
        end_index = width * height;
    } else {
        // Yolo layer (Yolo v3)
        num_regions = static_cast<uint32_t>(mask_size);
        end_index = width * height * (classes + 1);
    }

    const uint32_t inputs_size = width * height * num_regions * (classes + coords + 1);
    for (unsigned int batch_idx = 0; batch_idx < batches; batch_idx++) {
        for (unsigned int n = 0; n < num_regions; n++) {
            int index = entry_index(width,
                                    height,
                                    coords,
                                    classes,
                                    inputs_size,
                                    batch_idx,
                                    n * width * height,
                                    0);
            std::transform(input + index,
                            input + index + 2 * width * height,
                            output + index,
                            [](T elem) { return sigmoid<T>(elem); });

            index = entry_index(width,
                                height,
                                coords,
                                classes,
                                inputs_size,
                                batch_idx,
                                n * width * height,
                                coords);
            std::transform(input + index,
                            input + index + end_index,
                            output + index,
                            [](T elem) { return sigmoid<T>(elem); });
        }
    }

    if (do_softmax) {
        int index = entry_index(width, height, coords, classes, inputs_size, 0, 0, coords + 1);
        int batch_offset = inputs_size / regions;
        for (unsigned int batch_idx = 0; batch_idx < batches * regions; batch_idx++) {
            softmax_generic<T>(input + index + batch_idx * batch_offset,
                                output + index + batch_idx * batch_offset,
                                1,
                                classes,
                                height,
                                width);
        }
    }
}

struct region_yolo_test_params {
    std::vector<uint32_t> tensor;
    std::vector<int64_t> mask;
    uint32_t coords;
    uint32_t classes;
    uint32_t regionNum;
    int32_t axis;
    int32_t end_axis;
    data_types dataType;
    format fmt;
    bool softMax;
};

template <typename T>
void runRegionTest(region_yolo_test_params& params, bool is_caching_test = false) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    const tensor kInputTensor(params.tensor[0], params.tensor[1], params.tensor[2], params.tensor[3]);
    auto inputData = rg.generate_random_1d<T>(params.tensor[0] * params.tensor[1] * params.tensor[2] * params.tensor[3], -1, 1);

    auto inputPrim = engine.allocate_memory({ params.dataType, format::bfyx, kInputTensor });
    set_values(inputPrim, inputData);

    topology topology;
    topology.add(input_layout("InputData", inputPrim->get_layout()));
    topology.add(reorder("reorder_pre", input_info("InputData"), params.fmt, params.dataType));
    topology.add(region_yolo("region_yolo", input_info("reorder_pre"), params.coords, params.classes,
                             params.regionNum, params.mask, static_cast<uint32_t>(params.mask.size()),
                             params.axis, params.end_axis, params.softMax));
    topology.add(reorder("reorder_post", input_info("region_yolo"), format::bfyx, params.dataType));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("InputData", inputPrim);

    auto outputs = network->execute();
    auto output = outputs.at("reorder_post").get_memory();
    cldnn::mem_lock<T> outputData(output, get_test_stream());

    /// reference value
    std::vector<T> refOutputData(inputData.size());
    region_yolo_ref<T>(inputData.data(), refOutputData.data(),
                       params.tensor, params.coords, params.classes,
                       params.regionNum, params.softMax, params.mask);

    /// compare values
    for (size_t i = 0; i < inputData.size(); ++i) {
        ASSERT_NEAR(refOutputData[i], outputData[i], 0.01);
    }
}
}  // namespace

TEST(region_yolo_gpu_fp32, bfyx) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f32, format::bfyx, false};
    runRegionTest<float>(params);
}

TEST(region_yolo_gpu_fp32, bfyx_softmax) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f32, format::bfyx, true};
    runRegionTest<float>(params);
}

TEST(region_yolo_gpu_fp32, byxf) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f32, format::byxf, false};
    runRegionTest<float>(params);
}

TEST(region_yolo_gpu_fp32, byxf_softmax) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f32, format::byxf, true};
    runRegionTest<float>(params);
}

TEST(region_yolo_gpu_fp16, bfyx) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f16, format::bfyx, false};
    runRegionTest<ov::float16>(params);
}

TEST(region_yolo_gpu_fp16, bfyx_softmax) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f16, format::bfyx, true};
    runRegionTest<ov::float16>(params);
}

TEST(region_yolo_gpu_fp16, byxf) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f16, format::byxf, false};
    runRegionTest<ov::float16>(params);
}

TEST(region_yolo_gpu_fp16, byxf_softmax) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f16, format::byxf, true};
    runRegionTest<ov::float16>(params);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(region_yolo_gpu_fp32, bfyx_cached) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f32, format::bfyx, false};
    runRegionTest<float>(params, true);
}

TEST(region_yolo_gpu_fp32, bfyx_softmax_cached) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f32, format::bfyx, true};
    runRegionTest<float>(params, true);
}

TEST(region_yolo_gpu_fp32, byxf_cached) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f32, format::byxf, false};
    runRegionTest<float>(params, true);
}

TEST(region_yolo_gpu_fp32, byxf_softmax_cached) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f32, format::byxf, true};
    runRegionTest<float>(params, true);
}

TEST(region_yolo_gpu_fp16, bfyx_cached) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f16, format::bfyx, false};
    runRegionTest<ov::float16>(params, true);
}

TEST(region_yolo_gpu_fp16, bfyx_softmax_cached) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f16, format::bfyx, true};
    runRegionTest<ov::float16>(params, true);
}

TEST(region_yolo_gpu_fp16, byxf_cached) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f16, format::byxf, false};
    runRegionTest<ov::float16>(params, true);
}
#endif  // RUN_ALL_MODEL_CACHING_TESTS
TEST(region_yolo_gpu_fp16, byxf_softmax_cached) {
    region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, 1, 3, data_types::f16, format::byxf, true};
    runRegionTest<ov::float16>(params, true);
}
