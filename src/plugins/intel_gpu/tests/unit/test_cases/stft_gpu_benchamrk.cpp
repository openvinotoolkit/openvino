// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/stft.hpp>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

// PK: TEMPORARY BENCHMARK, WILL BE REMOVED BEFORE MERGING TO MASTER.

namespace {

const int WARMUPS = 10;
const int RUNS = 100;

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

}  // namespace helpers

struct STFTTestParams {
    ov::PartialShape signalShape;
    ov::PartialShape windowShape;
    ov::PartialShape outputShape;
    int64_t frameSize;
    int64_t frameStep;
    bool transposedFrames;
    std::vector<float> signalData;
    std::vector<float> windowData;
    std::vector<float> expectedOutput;
    std::string testcaseName;
};

class stft_benchmark : public ::testing::Test {
public:
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
        ret.frameStep = helpers::AllocateTensor<int64_t>({}, {testParam.frameStep});
        ret.frameSize = helpers::AllocateTensor<int64_t>({}, {testParam.frameSize});

        return ret;
    }

    void Execute(const STFTInferenceParams& params) {
        // Prepare the network.

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

        auto stream = get_test_stream_ptr(get_test_default_config(engine_));
        cldnn::network::ptr network = get_network(engine_, topology, get_test_default_config(engine_), stream, false);

        network->set_input_data("signal", params.signal);
        network->set_input_data("window", params.window);
        network->set_input_data("frameSize", params.frameSize);
        network->set_input_data("frameStep", params.frameStep);

        // Run and check results.
        const int warmup = WARMUPS;
        const int run = RUNS;

        std::map<primitive_id, network_output> outputs;
        for (int i = 0; i < warmup; ++i)
            network->execute();
        network->reset_execution(true);

        // Note: Should be based on events, this one
        // also adds up kernel launch time and gpu idle time.
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < run; ++i)
            network->execute();
        network->reset_execution(true);
        auto stop = std::chrono::system_clock::now();

        const auto d_actual = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

        outputs = network->execute();
        auto output = outputs.at("stft").get_memory();
        auto outputShape = output->get_layout().get_shape();
        std::cout << "Avg Time for output shape " << outputShape << ":" << d_actual / run << " microseconds\n\n";
    }

    template <ov::element::Type_t TYPE>
    void RunBenchmark(const ov::PartialShape& signalShape, int frameSize, int frameStep, bool transposed) {
        std::cout << "Benchmark: signal shape: " << signalShape << ", frameSize: " << frameSize
                  << ", frameStep: " << frameStep << ", transposed: " << transposed << std::endl;
        struct STFTTestParams params;
        params.signalShape = signalShape;
        params.windowShape = {frameSize};
        params.frameSize = frameSize;
        params.frameStep = frameStep;
        params.transposedFrames = transposed;
        params.signalData = std::vector<float>(ov::shape_size(params.signalShape.get_shape()), 0);
        params.windowData = std::vector<float>(ov::shape_size(params.windowShape.get_shape()), 0);
        params.testcaseName = "";

        Execute(PrepareInferenceParams<TYPE>(params));
    }

private:
    engine& engine_ = get_test_engine();
};
}  // namespace

TEST_F(stft_benchmark, DISABLED_benchmarks) {
    RunBenchmark<ov::element::Type_t::f32>({10000}, 1000, 2, true);
    RunBenchmark<ov::element::Type_t::f32>({10000}, 1000, 2, false);

    RunBenchmark<ov::element::Type_t::f32>({32768}, 2048, 512, true);
    RunBenchmark<ov::element::Type_t::f32>({32768}, 2048, 512, false);

    RunBenchmark<ov::element::Type_t::f32>({10000}, 100, 2, true);
    RunBenchmark<ov::element::Type_t::f32>({10000}, 100, 2, false);

    RunBenchmark<ov::element::Type_t::f32>({10000}, 1000, 200, true);
    RunBenchmark<ov::element::Type_t::f32>({10000}, 1000, 200, false);
}
