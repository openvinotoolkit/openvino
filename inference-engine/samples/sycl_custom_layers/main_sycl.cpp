#include "sycl_ext.hpp"

int main (int argc, char** argv) {
    std::vector<size_t> inpShape{1, 3, 10, 11};
    std::vector<size_t> outShape{1, 3, 10, 11};

    // Build network topology
    std::shared_ptr<ngraph::Node> input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape(inpShape));
    std::shared_ptr<ngraph::Node> relu1 = std::make_shared<ngraph::op::Relu>(input);
    std::shared_ptr<ngraph::Node> pool =  std::make_shared<ngraph::op::v1::MaxPool>(relu1,
                                                                                    ngraph::Strides({1, 1}),  // strides
                                                                                    ngraph::Shape({0, 0}),    // pads_begin
                                                                                    ngraph::Shape({0, 0}),    // pads_end
                                                                                    ngraph::Shape({1, 1}),    // kernel
                                                                                    ngraph::op::RoundingType::FLOOR,
                                                                                    ngraph::op::PadType::VALID);
    std::shared_ptr<ngraph::Node> relu = std::make_shared<ngraph::op::Relu>(pool);
    std::shared_ptr<ngraph::Node> ocl_layer = std::make_shared<SYCLLayerOp>(relu);

    auto ngraph_function = std::make_shared<ngraph::Function>(
            ocl_layer, ngraph::ParameterVector{std::dynamic_pointer_cast<ngraph::op::Parameter>(input)});

    // Load network
    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<InfEngineNgraphExtension>(), "GPU");
    InferenceEngine::CNNNetwork net(ngraph_function);

    // Create executable network with SYCL engine
    using namespace InferenceEngine::PluginConfigParams;
    InferenceEngine::ExecutableNetwork execNet = ie.LoadNetwork(net, "GPU");
//            {{InferenceEngine::PluginConfigParams::KEY_GPU_ENGINE_TYPE,
//              InferenceEngine::PluginConfigParams::KEY_GPU_SYCL_ENGINE}});
    InferenceEngine::InferRequest infRequest = execNet.CreateInferRequest();

    // Run inference
    std::cout << "Input shape: " << inpShape[0] << "," << inpShape[1] << "," << inpShape[2] << "," << inpShape[3] << std::endl;
    std::cout << "Output shape: " << outShape[0] << "," << outShape[1] << "," << outShape[2] << "," << outShape[3] << std::endl;

    std::vector<float> inpData(inpShape[0] * inpShape[1] * inpShape[2] * inpShape[3], 0);
    std::vector<float> outData(inpShape[0] * inpShape[1] * inpShape[2] * inpShape[3], 0);

    InferenceEngine::BlobMap inputBlobs, outputBlobs;
    inputBlobs[net.getInputsInfo().begin()->first] = InferenceEngine::make_shared_blob<float>({
        InferenceEngine::Precision::FP32,
        inpShape,
        InferenceEngine::Layout::ANY}, inpData.data());
    outputBlobs[net.getOutputsInfo().begin()->first] = InferenceEngine::make_shared_blob<float>({
        InferenceEngine::Precision::FP32,
        outShape,
        InferenceEngine::Layout::ANY}, outData.data());

    infRequest.SetInput(inputBlobs);
    infRequest.SetOutput(outputBlobs);
    infRequest.Infer();

    return 0;
}
