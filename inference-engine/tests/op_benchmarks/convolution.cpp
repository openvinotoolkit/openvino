#include "ngraph/runtime/reference/convolution.hpp"

#include <benchmark/benchmark.h>

#include <ie_core.hpp>
#include <numeric>
#include <vector>

#include "ngraph/ngraph.hpp"

using namespace ngraph;

namespace {
enum class DeviceType { CPU, TEMPLATE };
template <typename T>
struct BenchmarkConfig {
    Strides strides;
    CoordinateDiff pads_begin;
    CoordinateDiff pads_end;
    Strides dilations;
    Shape inputs_shape;
    std::vector<T> inputs;
    Shape filter_shape;
    std::vector<T> filters;
    Shape outputs_shape;
    std::vector<T> outputs;
    op::PadType auto_pad;
};

template <typename T>
std::shared_ptr<ngraph::Function> getFunction(const BenchmarkConfig<T>& cfg) {
    auto inputs_param =
        std::make_shared<op::Parameter>(element::f32, cfg.inputs_shape);
    inputs_param->set_friendly_name("inputs");
    auto filters_param =
        std::make_shared<op::Parameter>(element::f32, cfg.filter_shape);
    filters_param->set_friendly_name("filters");
    auto conv = std::make_shared<op::v1::Convolution>(
        inputs_param, filters_param, cfg.strides, cfg.pads_begin, cfg.pads_end,
        cfg.dilations, cfg.auto_pad);
    conv->set_friendly_name("convolution");
    return std::make_shared<Function>(
        conv, ParameterVector{inputs_param, filters_param});
}

BenchmarkConfig<float> getConfig() {
    using T = float;

    BenchmarkConfig<T> cfg;
    cfg.strides = Strides{1, 1};
    cfg.pads_begin = cfg.pads_end = CoordinateDiff{0, 0};
    cfg.dilations = Strides{1, 1};
    cfg.auto_pad = op::PadType::EXPLICIT;

#if 1
    cfg.inputs_shape = Shape{1, 1, 512, 512};
    cfg.inputs = std::vector<T>(shape_size(cfg.inputs_shape));
    std::iota(cfg.inputs.begin(), cfg.inputs.end(), 0);
    cfg.outputs_shape =
        Shape{1, 1, cfg.inputs_shape[2] - 2, cfg.inputs_shape[3] - 2};
    cfg.outputs = std::vector<T>(shape_size(cfg.outputs_shape));
#else
    cfg.inputs_shape = Shape{1, 1, 4, 4};
    cfg.inputs = std::vector<T>{1.0f, 3.0f, 5.0f, 7.0f, 7.0f, 5.0f, 3.0f, 1.0f,
                                2.0f, 4.0f, 6.0f, 8.0f, 8.0f, 6.0f, 4.0f, 2.0f};
    cfg.outputs_shape = Shape{1, 1, 2, 2};
    cfg.outputs = std::vector<T>{47.0f, 69.0f, 70.0f, 48.0f};
#endif
    cfg.filter_shape = Shape{1, 1, 3, 3};
    cfg.filters =
        std::vector<T>{1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 0.0f, 3.0f, 2.0f, 1.0f};

    return cfg;
}
}  // namespace

// benchmark for pure runtime::reference::convolution()
// (can be written if there is no evaluate())
static void convolution_2D_reference(benchmark::State& state) {
    // setup
    const auto cfg = getConfig();
    std::vector<float> out(shape_size(cfg.outputs_shape));

    // benchmark
    for (auto _ : state) {
        benchmark::DoNotOptimize(out.data());
        runtime::reference::convolution(
            cfg.inputs.data(), cfg.filters.data(), out.data(), cfg.inputs_shape,
            cfg.filter_shape, cfg.outputs_shape, cfg.strides, cfg.dilations,
            cfg.pads_begin, cfg.pads_end, cfg.strides);

        benchmark::ClobberMemory();
    }

// sanity check: generated output should be the same as expected
#if 0
    for (size_t i = 0; i < cfg.outputs.size(); i++) {
        if (cfg.outputs[i] != out[i]) {
            std::cout << "convolution_2D_reference failed!" << std::endl;
            exit(-1);
        }
    }
#endif
}
BENCHMARK(convolution_2D_reference)->Unit(benchmark::kMicrosecond);

// benchmark for InferenceEngine::InferRequest::Infer()
// (can be used for plugins & reference if it has evaluate() via TEMPLATE plugin)
template <DeviceType dev>
static void convolution_2D_plugin(benchmark::State& state) {
    using namespace InferenceEngine;

    // setup
    const auto cfg = getConfig();
    const auto f = getFunction(cfg);

    const auto network = InferenceEngine::CNNNetwork(f);
    InferenceEngine::Core ie;
    auto exe_network = dev == DeviceType::TEMPLATE
                           ? ie.LoadNetwork(network, "TEMPLATE")
                           : ie.LoadNetwork(network, "CPU");
    auto inference_req = exe_network.CreateInferRequest();

    {
        const auto inputs_info = network.getInputsInfo();

        const auto in = InferenceEngine::make_shared_blob<float>(
            {InferenceEngine::Precision::FP32, cfg.inputs_shape,
             InferenceEngine::Layout::NCHW},
            const_cast<float*>(cfg.inputs.data()), cfg.inputs.size());
        inference_req.SetBlob("inputs", in);

        const auto filters = InferenceEngine::make_shared_blob<float>(
            {InferenceEngine::Precision::FP32, cfg.filter_shape,
             InferenceEngine::Layout::NCHW},
            const_cast<float*>(cfg.filters.data()), cfg.filters.size());
        inference_req.SetBlob("filters", filters);
    }

    {
        DataPtr output_info = network.getOutputsInfo()["convolution"];
        output_info->setPrecision(InferenceEngine::Precision::FP32);
    }

    // benchmark
    for (auto _ : state) {
        inference_req.Infer();
    }

// sanity check: generated output should be the same as expected
#if 0   
    auto output_blob = inference_req.GetBlob("convolution");
    auto const memLocker = output_blob->cbuffer();  // use const memory locker
    const float* out = memLocker.as<const float*>();
     
    for (size_t i = 0; i < cfg.outputs.size(); i++) {
        if (cfg.outputs[i] != out[i]) {
            std::cout << "convolution_2D_template_plugin failed!" << std::endl;
            exit(-1);
        }
    }
#endif
}

BENCHMARK_TEMPLATE(convolution_2D_plugin, DeviceType::TEMPLATE)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(convolution_2D_plugin, DeviceType::CPU)
    ->Unit(benchmark::kMicrosecond);
