// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/region_yolo.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/runtime/memory.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

namespace internal
{
    static inline int entry_index(int width,
                                    int height,
                                    int coords,
                                    int classes,
                                    int outputs,
                                    int batch,
                                    int location,
                                    int entry)
    {
        int n = location / (width * height);
        int loc = location % (width * height);
        return batch * outputs + n * width * height * (coords + classes + 1) +
                entry * width * height + loc;
    }

    template <typename T>
    static inline T sigmoid(float x)
    {
        return static_cast<T>(1.f / (1.f + std::exp(-x)));
    }

    template <typename T>
    static inline void softmax_generic(const T* src_data, T* dst_data,
        uint32_t batches, uint32_t channels, uint32_t height, uint32_t width)
    {
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

    uint32_t shape_size(const std::vector<uint32_t>& input_shape)
    {
        uint32_t ret = 1;
        std::for_each(input_shape.begin(), input_shape.end(), [&ret](uint32_t n){
            ret *= n;
        });

        return ret;
    }

    template <typename T>
    void region_yolo(const T* input,
                        T* output,
                        const std::vector<uint32_t>& input_shape,
                        const uint32_t coords,
                        const uint32_t classes,
                        const uint32_t regions,
                        const bool do_softmax,
                        const std::vector<int64_t>& mask)
    {
        EXPECT_EQ(input_shape.size(), 4);

        const uint32_t batches = input_shape[0];
        //const uint32_t channels = input_shape[1];
        const uint32_t height = input_shape[2];
        const uint32_t width = input_shape[3];

        const auto mask_size = mask.size();

        std::copy(input, input + shape_size(input_shape), output);

        uint32_t num_regions = 0;
        uint32_t end_index = 0;

        if (do_softmax)
        {
            // Region layer (Yolo v2)
            num_regions = regions;
            end_index = width * height;
        }
        else
        {
            // Yolo layer (Yolo v3)
            num_regions = static_cast<uint32_t>(mask_size);
            end_index = width * height * (classes + 1);
        }

        const uint32_t inputs_size = width * height * num_regions * (classes + coords + 1);
        for (unsigned int batch_idx = 0; batch_idx < batches; batch_idx++)
        {
            for (unsigned int n = 0; n < num_regions; n++)
            {
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

        if (do_softmax)
        {
            int index =
                entry_index(width, height, coords, classes, inputs_size, 0, 0, coords + 1);
            int batch_offset = inputs_size / regions;
            for (unsigned int batch_idx = 0; batch_idx < batches * regions; batch_idx++)
            {
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
        data_types dataType;
        format fmt;
        bool softMax;
    };
}

template <typename T>
static void runRegionTest(internal::region_yolo_test_params& params)
{
    auto& engine = get_test_engine();
    const tensor kInputTensor(params.tensor[0], params.tensor[1], params.tensor[2], params.tensor[3]);
    auto inputData = generate_random_1d<T>(params.tensor[0] * params.tensor[1] * params.tensor[2] * params.tensor[3], -1, 1);

    auto inputPrim = engine.allocate_memory({ params.dataType, format::bfyx, kInputTensor });
    set_values(inputPrim, inputData);

    topology topology;
    topology.add(input_layout("InputData", inputPrim->get_layout()));
    topology.add(reorder("reorder_pre", "InputData", params.fmt, params.dataType));
    topology.add(region_yolo("region_yolo", "reorder_pre", params.coords, params.classes,
            params.regionNum, static_cast<uint32_t>(params.mask.size()), params.softMax));
    topology.add(reorder("reorder_post", "region_yolo", format::bfyx, params.dataType));

    network network(engine, topology);
    network.set_input_data("InputData", inputPrim);

    auto outputs = network.execute();
    auto output = outputs.at("reorder_post").get_memory();
    cldnn::mem_lock<T> outputData(output, get_test_stream());

    /// reference value
    std::vector<T> refOutputData(inputData.size());
    internal::region_yolo<T>(inputData.data(), refOutputData.data(),
                             params.tensor, params.coords, params.classes,
                             params.regionNum, params.softMax, params.mask);

    /// compare values
    for (size_t i = 0; i < inputData.size(); ++i) {
        EXPECT_NEAR(refOutputData[i], outputData[i], 0.01);
    }
}

TEST(region_yolo_gpu_fp32, bfyx) {
    internal::region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, data_types::f32, format::bfyx, false};
    runRegionTest<float>(params);
}

TEST(region_yolo_gpu_fp32, bfyx_softmax) {
    internal::region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, data_types::f32, format::bfyx, true};
    runRegionTest<float>(params);
}

TEST(region_yolo_gpu_fp32, byxf) {
    internal::region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, data_types::f32, format::byxf, false};
    runRegionTest<float>(params);
}

TEST(region_yolo_gpu_fp32, byxf_softmax) {
    internal::region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, data_types::f32, format::byxf, true};
    runRegionTest<float>(params);
}

TEST(region_yolo_gpu_fp16, bfyx) {
    internal::region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, data_types::f16, format::bfyx, false};
    runRegionTest<FLOAT16>(params);
}

TEST(region_yolo_gpu_fp16, bfyx_softmax) {
    internal::region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, data_types::f16, format::bfyx, true};
    runRegionTest<FLOAT16>(params);
}

TEST(region_yolo_gpu_fp16, byxf) {
    internal::region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, data_types::f16, format::byxf, false};
    runRegionTest<FLOAT16>(params);
}

TEST(region_yolo_gpu_fp16, byxf_softmax) {
    internal::region_yolo_test_params params{{ 1, 33, 52, 52 }, { 0, 1, 2 }, 4, 6, 3, data_types::f16, format::byxf, true};
    runRegionTest<FLOAT16>(params);
}
