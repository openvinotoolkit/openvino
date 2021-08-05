//#include <CL/sycl.hpp>
#include "sycl_ext.hpp"
#include <gpu/gpu_context_api_ocl.hpp>

#include <regex>

// nGraph op
NGRAPH_RTTI_DEFINITION(SYCLLayerOp, "SYCLLayerOp", 0);

SYCLLayerOp::SYCLLayerOp(const ngraph::Output<ngraph::Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

void SYCLLayerOp::validate_and_infer_types() {
    // Operation doesn't change shapes and element type
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

inline std::shared_ptr<ngraph::Node> getNodeByName(const std::string& name,
                                                   const std::shared_ptr<ngraph::Function> func,
                                                   bool namePattern = true) {
    for (const auto& node : func->get_ops()) {
        bool match = namePattern ? node->get_friendly_name().find(name) == 0 :
                     node->get_friendly_name() == name;
        if (match) {
            return node;
        }
    }
    std::cout << "Could not find a node: " + name << std::endl;
    return nullptr;
}

std::shared_ptr<ngraph::Node> SYCLLayerOp::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SYCLLayerOp>(new_args.at(0));
}

bool SYCLLayerOp::visit_attributes(ngraph::AttributeVisitor& visitor) {
    return true;
}

std::vector<std::string> InfEngineNgraphExtension::getImplTypes(const std::shared_ptr<ngraph::Node>& node) {
    return {"GPU"};
}

InferenceEngine::ILayerImpl::Ptr
InfEngineNgraphExtension::getImplementation(const std::shared_ptr<ngraph::Node>& node,
                                            const std::string& implType) {
    if (std::dynamic_pointer_cast<SYCLLayerOp>(node)) {
        return std::make_shared<SYCLLayerImpl>(node);
    }
    return nullptr;
}

SYCLLayerImpl::SYCLLayerImpl(const std::shared_ptr<ngraph::Node>& node)
{
    inpShapes.resize(node->get_input_size());
    for (size_t i = 0; i < inpShapes.size(); ++i)
        inpShapes[i] = node->get_input_shape(i);
    outShape = node->get_output_shape(0);
}

InferenceEngine::StatusCode SYCLLayerImpl::init(InferenceEngine::LayerConfig& config,
                                    InferenceEngine::ResponseDesc *resp) noexcept
{
    return InferenceEngine::StatusCode::OK;
}

InferenceEngine::StatusCode
SYCLLayerImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                                          InferenceEngine::ResponseDesc* resp) noexcept
{
    std::vector<InferenceEngine::DataConfig> inDataConfig;
    std::vector<InferenceEngine::DataConfig> outDataConfig;

    // Allow any offset before data
    size_t offset((std::numeric_limits<size_t>::max)());

    // Input shape
    for (const auto& shape : inpShapes)
    {
        InferenceEngine::SizeVector order(shape.size());
        std::iota(order.begin(), order.end(), 0);

        InferenceEngine::DataConfig inpConf;
        inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, shape, {shape, order, offset});

        // Input Remote Blob requirements
//        inpConf.allowedRemoteTypes = RemoteBlobTypes::SYCL; // can be OR'd together

        inDataConfig.push_back(inpConf);
    }

    // Output shape
    InferenceEngine::SizeVector order(outShape.size());
    std::iota(order.begin(), order.end(), 0);

    InferenceEngine::DataConfig outConf;
    outConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, outShape, {outShape, order, offset});
    // Output Remote Blob requirements
//    outConf.allowedRemoteTypes = RemoteBlobTypes::SYCL; // can be OR'd together

    outDataConfig.push_back(outConf);

    InferenceEngine::LayerConfig layerConfig;
    layerConfig.inConfs = inDataConfig;
    layerConfig.outConfs = outDataConfig;

    conf.push_back(layerConfig);
    return InferenceEngine::StatusCode::OK;
}

//class SYCLContext : public InferenceEngine::RemoteContext, public defails::param_map_obj_getter {
//public:
//    /**
//     * @brief A smart pointer to the ClContext object
//     */
//    using Ptr = std::shared_ptr<ClContext>;
//
//    /**
//     * @brief Returns the underlying OpenCL context handle.
//     */
//    sycl::queue getExecutionQueue() {
//        return _ObjFromParams<sycl::queue, gpu_handle_param>(getParams(), GPU_PARAM_KEY(SYCL_QUEUE),
//                GPU_PARAM_KEY(CONTEXT_TYPE), GPU_PARAM_VALUE(SYCL));
//    }
//};

InferenceEngine::StatusCode SYCLLayerImpl::execute(std::vector<InferenceEngine::Blob::Ptr>& inputs,
                                                   std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                                   InferenceEngine::ResponseDesc *resp) noexcept
{
    using InferenceEngine::gpu::USMBufferBlob;
    std::cout << "execute" << std::endl;

    // Note, we enforce SYCL input/outputs in LayerImpl::getSupportedConfigurations()
    auto inputRemoteBlob = inputs[0]->as<InferenceEngine::RemoteBlob>();
    assert(inputRemoteBlob != nullptr);
    auto outputRemoteBlob = outputs[0]->as<InferenceEngine::RemoteBlob>();
    assert(outputRemoteBlob != nullptr);

    // Note, we enforce SYCL input/outputs in LayerImpl::getSupportedConfigurations()
    auto inputBuffer = inputRemoteBlob->as<InferenceEngine::gpu::USMBufferBlob>();
    assert(inputBuffer != nullptr);
    auto outputBuffer = outputRemoteBlob->as<InferenceEngine::gpu::USMBufferBlob>();
    assert(inputBuffer != nullptr);

    float* in = static_cast<float*>(inputBuffer->get());
    float* out = static_cast<float*>(outputBuffer->get());

    // Implement operation on USM memory
    for(size_t idx = 0; idx < inputs[0]->size(); ++idx) {
        out[idx] = in[idx] + 3.0;
    };

//
//    // Get the SYCL queue from the remoteContext (don't even need the sycl context)
//    RemoteContext::Ptr remoteContext = inputSYCLBuffer->getContext();
//    sycl::queue syclQueue = remoteContext->as<InferenceEngine::SYCLContext>()->getExecutionQueue();
//
//    // Implement the operation with SYCL
//    syclQueue.submit([&](sycl::handler& h) {
//        sycl::accessor in(inputSYCLBuffer, h);
//        sycl::accessor out(outputSYCLBuffer, h);
//
//        h.parallel_for(inputs[0].size(), [=](sycl::item<1> idx) {
//            out[idx] = in[idx] + 3.0;
//        });
//    });

    // TODO - make this implicit, should be handled by the plugin
    // syclQueue.wait_and_throw();

    return InferenceEngine::OK;
}
