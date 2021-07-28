#include <ngraph/ngraph.hpp>
#include <inference_engine.hpp>

class SYCLLayerOp : public ngraph::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;

    SYCLLayerOp() = default;
    SYCLLayerOp(const ngraph::Output<ngraph::Node>& input);

    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
};

class SYCLLayerImpl : public InferenceEngine::ILayerExecImpl {
public:
    explicit SYCLLayerImpl(const std::shared_ptr<ngraph::Node>& node);

    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config,
                                     InferenceEngine::ResponseDesc *resp) noexcept override;

    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                                                           InferenceEngine::ResponseDesc* resp) noexcept override;

    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs,
                                        std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                        InferenceEngine::ResponseDesc *resp) noexcept override;

private:
    std::vector<ngraph::Shape> inpShapes;
    ngraph::Shape outShape;
};

class InfEngineNgraphExtension : public InferenceEngine::IExtension {
public:
    void Unload() noexcept override {}
    void Release() noexcept override { delete this; }
    void GetVersion(const InferenceEngine::Version*&) const noexcept override {}

    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override;

    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node,
                                                       const std::string& implType) override;
};
