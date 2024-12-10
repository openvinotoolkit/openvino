// Copyright (C) 2018-2024 Intel Corporation
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/stft.hpp>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

constexpr float EPS = 2e-3f;

namespace helpers {
// TODO: Move to common place.

// Converts float vector to another type vector.
template <typename T>
std::vector<T> ConverFloatVector(const std::vector<float>& vec) {
    std::vector<T> ret;
    ret.reserve(vec.size());
    for (const auto& val : vec) {
        ret.push_back(T(val));
    }
    return ret;
}

// Allocates tensoer with given shape and data.
template <typename TDataType>
memory::ptr AllocateTensor(ov::PartialShape shape, const std::vector<TDataType>& data) {
    const layout lo = {shape, ov::element::from<TDataType>(), cldnn::format::bfyx};
    EXPECT_EQ(lo.get_linear_size(), data.size());
    memory::ptr tensor = get_test_engine().allocate_memory(lo);
    set_values<TDataType>(tensor, data);
    return tensor;
}

template <typename T>
void CompareTypedBuffers(const memory::ptr& output, const memory::ptr& expectedOutput, cldnn::stream& stream) {
    mem_lock<T> output_ptr(output, stream);
    mem_lock<T> wanted_output_ptr(expectedOutput, stream);

    ASSERT_EQ(output->get_layout(), expectedOutput->get_layout());
    ASSERT_EQ(output_ptr.size(), wanted_output_ptr.size());
    for (size_t i = 0; i < output_ptr.size(); ++i)
        ASSERT_TRUE(are_equal(wanted_output_ptr[i], output_ptr[i], EPS)) << "at index " << i;
}

void CompareBuffers(const memory::ptr& output, const memory::ptr& expectedOutput, cldnn::stream& stream) {
    ASSERT_EQ(output->get_layout(), expectedOutput->get_layout());
    auto type = output->get_layout().data_type;

    switch (type) {
    case data_types::f32:
        helpers::CompareTypedBuffers<float>(output, expectedOutput, stream);
        break;

    default:
        ASSERT_TRUE(false) << "Unsupported data type: " << type;
        break;
    }
}

}  // namespace helpers

struct STFTTestParams {
    ov::PartialShape signalShape;
    ov::PartialShape windowShape;
    ov::PartialShape outputShape;
    bool transposedFrames;
    std::vector<float> signalData;
    std::vector<float> windowData;
    int64_t frameSize;
    int64_t frameStep;
    std::vector<float> expectedOutput;
    std::string testcaseName;
};

class stft_test : public ::testing::TestWithParam<STFTTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<STFTTestParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "signalShape=" << param.signalShape;
        result << "_windowShape=" << param.windowShape;
        result << "_outputShape=" << param.outputShape;
        result << "_frameSize=" << param.frameSize;
        result << "_frameStep=" << param.frameStep;
        result << "_transposedFrames=" << param.transposedFrames;
        result << "_" << param.testcaseName;
        return result.str();
    }

    struct STFTInferenceParams {
        bool transposedFrames;
        memory::ptr signal;
        memory::ptr window;
        memory::ptr frameSize;
        memory::ptr frameStep;
        memory::ptr expectedOutput;
    };

    template <ov::element::Type_t ET>
    STFTInferenceParams PrepareInferenceParams(const STFTTestParams& testParam) {
        using T = typename ov::element_type_traits<ET>::value_type;
        STFTInferenceParams ret;

        ret.transposedFrames = testParam.transposedFrames;

        ret.signal =
            helpers::AllocateTensor<T>(testParam.signalShape, helpers::ConverFloatVector<T>(testParam.signalData));
        ret.window =
            helpers::AllocateTensor<T>(testParam.windowShape, helpers::ConverFloatVector<T>(testParam.windowData));
        ret.expectedOutput =
            helpers::AllocateTensor<T>(testParam.outputShape, helpers::ConverFloatVector<T>(testParam.expectedOutput));

        ret.frameStep = helpers::AllocateTensor<int64_t>({}, {testParam.frameStep});
        ret.frameSize = helpers::AllocateTensor<int64_t>({}, {testParam.frameSize});

        return ret;
    }

    void Execute(const STFTInferenceParams& params) {
        // Prepare the network.
        auto stream = get_test_stream_ptr(get_test_default_config(engine_));

        auto scalar_layout = params.frameSize->get_layout();
        scalar_layout.set_partial_shape({});

        topology topology;
        topology.add(input_layout("signal", params.signal->get_layout()));
        topology.add(input_layout("window", params.window->get_layout()));
        topology.add(input_layout("frameSize", scalar_layout));
        topology.add(input_layout("frameStep", scalar_layout));
        topology.add(STFT("stft",
                          input_info("signal"),
                          input_info("window"),
                          input_info("frameSize"),
                          input_info("frameStep"),
                          params.transposedFrames));

        cldnn::network::ptr network = get_network(engine_, topology, get_test_default_config(engine_), stream, false);

        network->set_input_data("signal", params.signal);
        network->set_input_data("window", params.window);
        network->set_input_data("frameSize", params.frameSize);
        network->set_input_data("frameStep", params.frameStep);

        // Run and check results.
        auto outputs = network->execute();

        auto output = outputs.at("stft").get_memory();

        helpers::CompareBuffers(output, params.expectedOutput, get_test_stream());
    }

private:
    engine& engine_ = get_test_engine();
};

std::vector<STFTTestParams> generateTestParams() {
    std::vector<STFTTestParams> params;

    // params.emplace_back(signal_48,
    //                 hann_window_16,
    //                 frame_size_16,
    //                 frame_step_16,
    //                 transpose_frames_true,
    //                 output_9_3_2_transp,
    //                 "basic_1D_transp");

    params.push_back(STFTTestParams{
        {48},
        {16},
        {9, 3, 2},
        true,
        {-0.41676, -0.05627, -2.1362,  1.64027,  -1.79344, -0.84175, 0.50288,  -1.24529, -1.05795, -0.90901,
         0.55145,  2.29221,  0.04154,  -1.11793, 0.53906,  -0.59616, -0.01913, 1.175,    -0.74787, 0.00903,
         -0.87811, -0.15643, 0.25657,  -0.98878, -0.33882, -0.23618, -0.63766, -1.18761, -1.42122, -0.1535,
         -0.26906, 2.23137,  -2.43477, 0.11273,  0.37044,  1.35963,  0.50186,  -0.84421, 0.00001,  0.54235,
         -0.31351, 0.77101,  -1.86809, 1.73118,  1.46768,  -0.33568, 0.61134,  0.04797},
        {0.,
         0.04323,
         0.16543,
         0.34549,
         0.55226,
         0.75,
         0.90451,
         0.98907,
         0.98907,
         0.90451,
         0.75,
         0.55226,
         0.34549,
         0.16543,
         0.04323,
         0.},
        16,
        16,
        {-2.52411, 0.,       -3.6289,  0.,      1.1366,   0.,       1.99743,  2.45799,  1.84867,  -0.67991, 0.26235,
         0.25725,  -2.243,   -1.74288, 0.39666, 0.60667,  -0.73965, -0.24622, 2.91255,  -0.82545, 0.03844,  0.45931,
         -1.29728, -1.50822, -2.56084, 2.24181, -0.92956, -1.32518, 1.78749,  1.94867,  0.87525,  0.70978,  0.47508,
         1.29318,  -0.18799, 0.98232,  2.10241, -2.57882, 0.88504,  -1.03814, -1.44897, -2.97866, -1.59965, -0.02599,
         -1.02171, 0.17824,  2.46326,  1.82815, -0.44417, 0.,       0.24368,  0.,       -2.81501, 0.},
        "basic_1D_transp"});
    return params;
}

}  // namespace

#define STFT_TEST_P(precision)                                                       \
    TEST_P(stft_test, ref_comp_##precision) {                                        \
        Execute(PrepareInferenceParams<ov::element::Type_t::precision>(GetParam())); \
    }

STFT_TEST_P(f32);

INSTANTIATE_TEST_SUITE_P(stft_test_suit,
                         stft_test,
                         testing::ValuesIn(generateTestParams()),
                         stft_test::getTestCaseName);
