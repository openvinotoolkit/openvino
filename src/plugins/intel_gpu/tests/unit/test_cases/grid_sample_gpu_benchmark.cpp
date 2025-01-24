// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>

#include "intel_gpu/primitives/grid_sample.hpp"
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

struct GridSampleTestParams {
    ov::PartialShape inputShape;
    ov::PartialShape gridShape;
    ov::PartialShape outputShape;
    GridSampleOp::Attributes attributes;
    std::vector<float> inputData;
    std::vector<float> gridData;
};

class gridSample_benchmark : public ::testing::Test {
public:
    struct GridSamplInferenceParams {
        GridSampleOp::Attributes attributes;
        memory::ptr input;
        memory::ptr grid;
        memory::ptr expectedOutput;
    };

    template <ov::element::Type_t ET>
    GridSamplInferenceParams PrepareInferenceParams(const GridSampleTestParams& testParam) {
        using T = typename ov::element_type_traits<ET>::value_type;
        GridSamplInferenceParams ret;

        ret.attributes = testParam.attributes;

        ret.input =
            helpers::AllocateTensor<T>(testParam.inputShape, helpers::ConverFloatVector<T>(testParam.inputData));
        ret.grid = helpers::AllocateTensor<T>(testParam.gridShape, helpers::ConverFloatVector<T>(testParam.gridData));
        return ret;
    }

    void Execute(const GridSamplInferenceParams& params) {
        // Prepare the network.

        topology topology;
        topology.add(input_layout("input", params.input->get_layout()));
        topology.add(input_layout("grid", params.grid->get_layout()));
        topology.add(grid_sample("grid_sample", {input_info("input"), input_info("grid")}, params.attributes));

        auto stream = get_test_stream_ptr(get_test_default_config(engine_));
        cldnn::network::ptr network = get_network(engine_, topology, get_test_default_config(engine_), stream, false);

        network->set_input_data("input", params.input);
        network->set_input_data("grid", params.grid);

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
        auto output = outputs.at("grid_sample").get_memory();
        auto outputShape = output->get_layout().get_shape();
        std::cout << "Avg Time for output shape " << outputShape << ":" << d_actual / run << " microseconds\n\n";
    }

    template <ov::element::Type_t TYPE>
    void RunBenchmark(const ov::PartialShape& inputShape, const ov::PartialShape& gridShape) {
        std::cout << "Benchmark: input shape: " << inputShape << ", grid shape: " << gridShape << std::endl;
        struct GridSampleTestParams params;
        params.inputShape = inputShape;
        params.gridShape = gridShape;
        params.inputData = std::vector<float>(ov::shape_size(params.inputShape.get_shape()), 0);
        params.gridData = std::vector<float>(ov::shape_size(params.gridShape.get_shape()), 0);
        params.attributes.mode = GridSampleOp::InterpolationMode::BILINEAR;
        params.attributes.align_corners = true;
        params.attributes.padding_mode = GridSampleOp::PaddingMode::ZEROS;
        Execute(PrepareInferenceParams<TYPE>(params));
    }

private:
    engine& engine_ = get_test_engine();
};
}  // namespace

TEST_F(gridSample_benchmark, benchmarks) {
    RunBenchmark<ov::element::Type_t::f32>({1, 128, 120, 216}, {1, 120, 216, 2});
}