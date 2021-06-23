#include "ngraph/runtime/reference/convolution.hpp"

#include <benchmark/benchmark.h>

#include <ie_core.hpp>
#include <numeric>
#include <vector>

#include "ngraph/ngraph.hpp"

using namespace ngraph;

namespace {
// helper type needed to use templated benchmarks feature
enum class DeviceType { CPU, TEMPLATE };

// description of Convolution input/output tensors & attributes
template <typename T>
struct ConvolutionConfiguration {
    Strides strides;
    CoordinateDiff pads_begin;
    CoordinateDiff pads_end;
    Strides dilations;
    Shape inputs_shape;
    std::vector<T> inputs;
    Shape filter_shape;
    std::vector<T> filters;
    Shape outputs_shape;
    op::PadType auto_pad;
};

// creates ngraph::Function with single Convolution node
template <typename T>
std::shared_ptr<ngraph::Function> ConvolutionModel(
    const ConvolutionConfiguration<T>& cfg) {
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

// creates configuration used in ConvolutionModel() benchmarking
ConvolutionConfiguration<float> ConvolutionModelConfiguration() {
    using T = float;

    // specific configuration for benchmarks bellow
    ConvolutionConfiguration<T> cfg;
    cfg.strides = Strides{1, 1};
    cfg.pads_begin = cfg.pads_end = CoordinateDiff{0, 0};
    cfg.dilations = Strides{1, 1};
    cfg.auto_pad = op::PadType::EXPLICIT;
    cfg.inputs_shape = Shape{1, 1, 512, 512};
    cfg.inputs = std::vector<T>(shape_size(cfg.inputs_shape));
    std::iota(cfg.inputs.begin(), cfg.inputs.end(), 0);
    cfg.filter_shape = Shape{1, 1, 3, 3};
    cfg.filters =
        std::vector<T>{1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 0.0f, 3.0f, 2.0f, 1.0f};
    cfg.outputs_shape =
        Shape{1, 1, cfg.inputs_shape[2] - 2, cfg.inputs_shape[3] - 2};

    return cfg;
}

}  // namespace

// benchmark for pure runtime::reference::convolution()
// (can be written if there is no evaluate())
static void convolution_2D_reference(benchmark::State& state) {
    // setup
    const auto cfg = ConvolutionModelConfiguration();
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
}
BENCHMARK(convolution_2D_reference)->Unit(benchmark::kMicrosecond);

// benchmark for InferenceEngine::InferRequest::Infer()
// (can be used for plugins & reference if it has evaluate() via TEMPLATE
// plugin)
template <DeviceType dev>
static void convolution_2D_plugin(benchmark::State& state) {
    // setup
    const auto cfg = ConvolutionModelConfiguration();
    const auto f = ConvolutionModel(cfg);
    const auto network = InferenceEngine::CNNNetwork(f);
    InferenceEngine::Core ie;
    auto exe_network = (dev == DeviceType::TEMPLATE)
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
        InferenceEngine::DataPtr output_info =
            network.getOutputsInfo()["convolution"];
        output_info->setPrecision(InferenceEngine::Precision::FP32);
    }

    // benchmark
    for (auto _ : state) {
        inference_req.Infer();
    }
}
BENCHMARK_TEMPLATE(convolution_2D_plugin, DeviceType::TEMPLATE)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(convolution_2D_plugin, DeviceType::CPU)
    ->Unit(benchmark::kMicrosecond);
