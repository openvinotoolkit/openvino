// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/reorg_yolo.hpp>

#include <cstddef>
#include <string>

using namespace cldnn;
using namespace ::tests;

namespace {
template<typename T>
struct ReorgYoloParams {
    ov::PartialShape inputTensor;
    std::vector<T> input;
    uint32_t stride;
    std::vector<T> expected;
};

template<typename T>
using ReorgYoloParamsWithLayout = std::tuple<
    ReorgYoloParams<T>,
    format::type,       // blocked layout
    bool                // should_fail
>;

const std::vector<format::type> dataFormats = {
    format::bfyx,
    format::yxfb,
    format::byxf,
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv16,
    format::bs_fs_yx_bsv32_fsv32
};

template<typename T>
std::vector<T> getValues(const std::vector<float> &values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template <typename T> float getError();

template<>
float getError<float>() {
    return 0.001;
}

template<>
float getError<ov::float16>() {
    return 0.2;
}

template<typename T>
std::vector<ReorgYoloParams<T>> generateParams() {
    static const std::vector<ReorgYoloParams<T>> result = {
        {
            ov::PartialShape{1, 4, 2, 2},
            getValues<T>({
                0.0, 1.0,
                2.0, 3.0,

                4.0, 5.0,
                6.0, 7.0,

                8.0, 9.0,
                10.0, 11.0,

                12.0, 13.0,
                14.0, 15.0,
            }),
            2,
            getValues<T>({
                0.0f, 2.0f, 8.0f, 10.0f,
                1.0f, 3.0f, 9.0f, 11.0f,

                4.0f, 6.0f, 12.0f, 14.0f,
                5.0f, 7.0f, 13.0f, 15.0f,
            })
        },
        {
            ov::PartialShape{2, 9, 3, 3},
            getValues<T>({
                0.0f, 1.0f, 2.0f,
                3.0f, 4.0f, 5.0f,
                6.0f, 7.0f, 8.0f,

                9.0f, 10.0f, 11.0f,
                12.0f, 13.0f, 14.0f,
                15.0f, 16.0f, 17.0f,

                18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f,
                24.0f, 25.0f, 26.0f,

                27.0f, 28.0f, 29.0f,
                30.0f, 31.0f, 32.0f,
                33.0f, 34.0f, 35.0f,

                36.0f, 37.0f, 38.0f, 39.0f,
                40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
                46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f,
                55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f,
                64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f,
                73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f, 81.0f,
                82.0f, 83.0f, 84.0f, 85.0f, 86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
                91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f,
                100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f,
                109.0f, 110.0f, 111.0f, 112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f,
                118.0f, 119.0f, 120.0f, 121.0f, 122.0f, 123.0f, 124.0f, 125.0f, 126.0f,
                127.0f, 128.0f, 129.0f, 130.0f, 131.0f, 132.0f, 133.0f, 134.0f, 135.0f,
                136.0f, 137.0f, 138.0f, 139.0f, 140.0f, 141.0f, 142.0f, 143.0f, 144.0f,
                145.0f, 146.0f, 147.0f, 148.0f, 149.0f, 150.0f, 151.0f, 152.0f, 153.0f,
                154.0f, 155.0f, 156.0f, 157.0f, 158.0f, 159.0f, 160.0f, 161.0f
            }),
            3,
            getValues<T>({
                0.0f, 3.0f, 6.0f, 27.0f, 30.0f, 33.0f, 54.0f, 57.0f, 60.0f,
                1.0f, 4.0f, 7.0f, 28.0f, 31.0f, 34.0f, 55.0f, 58.0f, 61.0f,
                2.0f, 5.0f, 8.0f, 29.0f, 32.0f, 35.0f, 56.0f, 59.0f, 62.0f,
                9.0f, 12.0f, 15.0f, 36.0f, 39.0f, 42.0f, 63.0f, 66.0f, 69.0f,
                10.0f, 13.0f, 16.0f, 37.0f, 40.0f, 43.0f, 64.0f, 67.0f, 70.0f,
                11.0f, 14.0f, 17.0f, 38.0f, 41.0f, 44.0f, 65.0f, 68.0f, 71.0f,
                18.0f, 21.0f, 24.0f, 45.0f, 48.0f, 51.0f, 72.0f, 75.0f, 78.0f,
                19.0f, 22.0f, 25.0f, 46.0f, 49.0f, 52.0f, 73.0f, 76.0f, 79.0f,
                20.0f, 23.0f, 26.0f, 47.0f, 50.0f, 53.0f, 74.0f, 77.0f, 80.0f,
                81.0f, 84.0f, 87.0f, 108.0f, 111.0f, 114.0f, 135.0f, 138.0f, 141.0f,
                82.0f, 85.0f, 88.0f, 109.0f, 112.0f, 115.0f, 136.0f, 139.0f, 142.0f,
                83.0f, 86.0f, 89.0f, 110.0f, 113.0f, 116.0f, 137.0f, 140.0f, 143.0f,
                90.0f, 93.0f, 96.0f, 117.0f, 120.0f, 123.0f, 144.0f, 147.0f, 150.0f,
                91.0f, 94.0f, 97.0f, 118.0f, 121.0f, 124.0f, 145.0f, 148.0f, 151.0f,
                92.0f, 95.0f, 98.0f, 119.0f, 122.0f, 125.0f, 146.0f, 149.0f, 152.0f,
                99.0f, 102.0f, 105.0f, 126.0f, 129.0f, 132.0f, 153.0f, 156.0f, 159.0f,
                100.0f, 103.0f, 106.0f, 127.0f, 130.0f, 133.0f, 154.0f, 157.0f, 160.0f,
                101.0f, 104.0f, 107.0f, 128.0f, 131.0f, 134.0f, 155.0f, 158.0f, 161.0f,
            }),
        },
        {
            ov::PartialShape{2, 5, 4, 4},
            getValues<T>({
                0.0f, 1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f, 7.0f,
                8.0f, 9.0f, 10.0f, 11.0f,
                12.0f, 13.0f, 14.0f, 15.0f,

                16.0f, 17.0f, 18.0f, 19.0f,
                20.0f, 21.0f, 22.0f, 23.0f,
                24.0f, 25.0f, 26.0f, 27.0f,
                28.0f, 29.0f, 30.0f, 31.0f,

                32.0f, 33.0f, 34.0f, 35.0f,
                36.0f, 37.0f, 38.0f, 39.0f,
                40.0f, 41.0f, 42.0f, 43.0f,
                44.0f, 45.0f, 46.0f, 47.0f,

                48.0f, 49.0f, 50.0f, 51.0f,
                52.0f, 53.0f, 54.0f, 55.0f,
                56.0f, 57.0f, 58.0f, 59.0f,
                60.0f, 61.0f, 62.0f, 63.0f,

                64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
                71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f,
                81.0f, 82.0f, 83.0f, 84.0f, 85.0f, 86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
                91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f, 100.0f,
                101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f, 110.0f,
                111.0f, 112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f, 120.0f,
                121.0f, 122.0f, 123.0f, 124.0f, 125.0f, 126.0f, 127.0f, 128.0f, 129.0f, 130.0f,
                131.0f, 132.0f, 133.0f, 134.0f, 135.0f, 136.0f, 137.0f, 138.0f, 139.0f, 140.0f,
                141.0f, 142.0f, 143.0f, 144.0f, 145.0f, 146.0f, 147.0f, 148.0f, 149.0f, 150.0f,
                151.0f, 152.0f, 153.0f, 154.0f, 155.0f, 156.0f, 157.0f, 158.0f, 159.0f,
            }),
            2,
            getValues<T>({
                0.0f, 2.0f,
                4.0f, 6.0f,

                16.0f, 18.0f,
                20.0f, 22.0f,

                32.0f, 34.0f,
                36.0f, 38.0f,

                48.0f, 50.0f,
                52.0f, 54.0f,

                1.0f, 3.0f,
                5.0f, 7.0f,

                17.0f, 19.0f,
                21.0f, 23.0f,

                33.0f, 35.0f,
                37.0f, 39.0f,

                49.0f, 51.0f,
                53.0f, 55.0f,

                8.0f, 10.0f, 12.0f, 14.0f, 24.0f, 26.0f, 28.0f, 30.0f,
                40.0f, 42.0f, 44.0f, 46.0f, 56.0f, 58.0f, 60.0f, 62.0f, 9.0f, 11.0f,
                13.0f, 15.0f, 25.0f, 27.0f, 29.0f, 31.0f, 41.0f, 43.0f, 45.0f, 47.0f,
                57.0f, 59.0f, 61.0f, 63.0f, 16.0f, 18.0f, 20.0f, 22.0f, 32.0f, 34.0f,
                36.0f, 38.0f, 48.0f, 50.0f, 52.0f, 54.0f, 64.0f, 66.0f, 68.0f, 70.0f,
                64.0f, 66.0f, 68.0f, 70.0f, 80.0f, 82.0f, 84.0f, 86.0f, 96.0f, 98.0f,
                100.0f, 102.0f, 112.0f, 114.0f, 116.0f, 118.0f, 65.0f, 67.0f, 69.0f, 71.0f,
                81.0f, 83.0f, 85.0f, 87.0f, 97.0f, 99.0f, 101.0f, 103.0f, 113.0f, 115.0f,
                117.0f, 119.0f, 72.0f, 74.0f, 76.0f, 78.0f, 88.0f, 90.0f, 92.0f, 94.0f,
                104.0f, 106.0f, 108.0f, 110.0f, 120.0f, 122.0f, 124.0f, 126.0f, 73.0f, 75.0f,
                77.0f, 79.0f, 89.0f, 91.0f, 93.0f, 95.0f, 105.0f, 107.0f, 109.0f, 111.0f,
                121.0f, 123.0f, 125.0f, 127.0f, 80.0f, 82.0f, 84.0f, 86.0f, 96.0f, 98.0f,
                100.0f, 102.0f, 112.0f, 114.0f, 116.0f, 118.0f, 128.0f, 130.0f, 132.0f, 134.0f,
            }),
        },
    };
    return result;
}

template<typename T>
std::vector<ReorgYoloParams<T>> generateInvalidParams() {
    static const std::vector<ReorgYoloParams<T>> result = {
        { // Feature < stride*stride
            ov::PartialShape{1, 3, 4, 4},
            getValues<T>({
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
            }),
            2,
            getValues<T>({}),
        },
        { // Height % stride != 0
            ov::PartialShape{1, 4, 5, 4},
            getValues<T>({
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
                51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f,
                61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
                71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
            }),
            2,
            getValues<T>({}),
        },
        { // Width % stride != 0
            ov::PartialShape{1, 4, 4, 5},
            getValues<T>({
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
                51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f,
                61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
                71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
            }),
            2,
            getValues<T>({}),
        },
    };
    return result;
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<ReorgYoloParamsWithLayout<T> > &param) {
        std::stringstream buf;
        ReorgYoloParams<T> p;
        format::type target_format;
        bool should_fail;
        std::tie(p, target_format, should_fail) = param.param;
        buf << "InputTensor=" << to_string(p.inputTensor)
            << ".stride=" << p.stride
            << ".TargetLayout=" << fmt_to_str(target_format);
        return buf.str();
    }
};
};  // namespace

template<typename T>
struct reorg_yolo_test
        : public ::testing::TestWithParam<ReorgYoloParamsWithLayout<T> > {
public:
    void test(bool is_caching_test) {
        ReorgYoloParams<T> params;
        format::type target_format;
        bool should_fail;
        std::tie(params, target_format, should_fail) = this->GetParam();

        if (should_fail) {
            ASSERT_ANY_THROW(run_test(params, target_format, is_caching_test));
        } else {
            ASSERT_NO_FATAL_FAILURE(run_test(params, target_format, is_caching_test));
        }
    }

private:
    void run_test(const ReorgYoloParams<T>& params, const format::type target_format, bool is_caching_test) {
        const auto data_type = ov::element::from<T>();
        const format::type plain_format = format::bfyx;

        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({params.inputTensor, data_type, plain_format});

        set_values(input, params.input);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(reorder("input_reordered", input_info("input"), target_format, data_type));
        topology.add(reorg_yolo("reorg_yolo", input_info("input_reordered"), params.stride));
        topology.add(reorder("reorg_yolo_reordered", input_info("reorg_yolo"), plain_format, data_type));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input);
        const auto result = network->execute();

        auto out_mem = result.at("reorg_yolo_reordered").get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.expected.size(), out_ptr.size());
        for (size_t i = 0; i < params.expected.size(); ++i) {
            ASSERT_NEAR(params.expected[i], out_ptr[i], getError<T>()) << "format=" << target_format << ", i= " << i;
        }
    }
};


using test_f32 = reorg_yolo_test<float>;
using test_f16 = reorg_yolo_test<ov::float16>;

TEST_P(test_f32, basic) {
    test(false);
}

TEST_P(test_f16, basic) {
    test(false);
}



INSTANTIATE_TEST_SUITE_P(reorg_yolo_f32,
                         test_f32,
                         ::testing::Combine(
                                 ::testing::Values(generateParams<float>()[0]),
                                 ::testing::ValuesIn(dataFormats),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(reorg_yolo_f16,
                         test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateParams<ov::float16>()),
                                 ::testing::ValuesIn(dataFormats),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(reorg_yolo_invalid_input,
                         test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateInvalidParams<float>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(true)),
                         PrintToStringParamName());

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(test_f32, basic_cached) {
    test(true);
}
#endif
TEST_P(test_f16, basic_cached) {
    test(true);
}
