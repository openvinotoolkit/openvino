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

static constexpr int WARMUPS = 10;
static constexpr int RUNS = 100;
static constexpr int SEED = 7877;

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

static std::vector<float> RandomFloatVector(size_t size, float low, float high) {
    std::vector<float> vec;
    vec.resize(size);
    static std::default_random_engine engine(SEED);
    std::uniform_real_distribution<float> dis(low, high);

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = dis(engine);
    }

    return vec;
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

        topology.add(reorder("reordered_data", input_info("input"), format::bfyx, data_types::f16));
        topology.add(reorder("reordered_grid", input_info("grid"), format::bfyx, data_types::f16));
        topology.add(grid_sample("grid_sample",
                                 {input_info("reordered_data"), input_info("reordered_grid")},
                                 params.attributes));
        topology.add(reorder("plane_grid_sample", input_info("grid_sample"), format::bfyx, data_types::f16));

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
        auto output = outputs.at("plane_grid_sample").get_memory();
        auto outputShape = output->get_layout().get_shape();
        std::cout << "Avg Time for output shape " << outputShape << ":" << d_actual / run << " microseconds\n\n";
    }

    GridSampleTestParams PrepareParams(const ov::PartialShape& inputShape,
                                       const std::vector<float>& inputData,
                                       const ov::PartialShape& gridShape,
                                       const std::vector<float>& gridData) {
        EXPECT_EQ(ov::shape_size(inputShape.get_shape()), inputData.size());
        EXPECT_EQ(ov::shape_size(gridShape.get_shape()), gridData.size());
        GridSampleTestParams params;
        params.inputShape = inputShape;
        params.gridShape = gridShape;
        params.inputData = inputData;
        params.gridData = gridData;
        params.attributes.mode = GridSampleOp::InterpolationMode::BILINEAR;
        params.attributes.align_corners = true;
        params.attributes.padding_mode = GridSampleOp::PaddingMode::ZEROS;
        return params;
    }

    GridSampleTestParams PrepareRandomDataParams(const ov::PartialShape& inputShape,
                                                 const ov::PartialShape& gridShape) {
        return PrepareParams(inputShape,
                             helpers::RandomFloatVector(ov::shape_size(inputShape.get_shape()), -1000.0f, 1000.0f),
                             gridShape,
                             helpers::RandomFloatVector(ov::shape_size(gridShape.get_shape()), -1.0f, 1.0f));
    }

    GridSampleTestParams PrepareGridDataStaticParams(const ov::PartialShape& inputShape,
                                                     const ov::PartialShape& gridShape) {
        return PrepareParams(inputShape,
                             std::vector<float>(ov::shape_size(inputShape.get_shape()), 0),
                             gridShape,
                             std::vector<float>(ov::shape_size(gridShape.get_shape()), 0));
    }

    GridSampleTestParams PrepareGridDataFileParams(const std::string& gridFilePath,
                                                   const ov::PartialShape& inputShape,
                                                   const ov::PartialShape& gridShape) {
        std::streampos gridFileSize;
        std::ifstream gridFile(gridFilePath, std::ios::binary);

        // get its size:
        gridFile.seekg(0, std::ios::end);
        gridFileSize = gridFile.tellg();
        gridFile.seekg(0, std::ios::beg);

        // read the data:
        std::vector<float> gridData(gridFileSize / sizeof(float));
        gridFile.read((char*)&gridData[0], gridFileSize);
        gridFile.close();

        return PrepareParams(inputShape,
                             helpers::RandomFloatVector(ov::shape_size(inputShape.get_shape()), -1000.0f, 1000.0f),
                             gridShape,
                             gridData);
    }

    template <ov::element::Type_t TYPE>
    void RunBenchmark(const std::string& name, const GridSampleTestParams& params) {
        std::cout << "Benchmark(" << name << "): input shape: " << params.inputShape
                  << ", grid shape: " << params.gridShape << std::endl;
        Execute(PrepareInferenceParams<TYPE>(params));
    }

private:
    engine& engine_ = get_test_engine();
};
}  // namespace

TEST_F(gridSample_benchmark, benchmarks) {
    RunBenchmark<ov::element::Type_t::f16>("random access", PrepareRandomDataParams({1, 128, 120, 216}, {1, 120, 216, 2}));
    RunBenchmark<ov::element::Type_t::f16>("constant access",
                                           PrepareGridDataStaticParams({1, 128, 120, 216}, {1, 120, 216, 2}));

    RunBenchmark<ov::element::Type_t::f16>("random access", PrepareRandomDataParams({2, 128, 80, 144}, {2, 11520, 81, 2}));
    RunBenchmark<ov::element::Type_t::f16>("constant access",
                                           PrepareGridDataStaticParams({2, 128, 80, 144}, {2, 11520, 81, 2}));

    RunBenchmark<ov::element::Type_t::f16>("access exported from real model",
                                           PrepareGridDataFileParams("src/plugins/intel_gpu/tests/unit/test_cases/data/grid.bin", {2, 128, 80, 144}, {2, 11520, 81, 2}));
}