// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <inference_engine.hpp>
#include <map>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <string>
#include <vector>

#define CUSTOM_RELU_TYPE "CustomReLU"

class CustomReLUImpl : public InferenceEngine::ILayerExecImpl {
public:
    explicit CustomReLUImpl(const std::shared_ptr<ngraph::Node>& node): _node(node) {}

    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                                                           InferenceEngine::ResponseDesc* /*resp*/) noexcept override {
        InferenceEngine::LayerConfig layerConfig;
        layerConfig.dynBatchSupport = true;

        if (_node->outputs().size() != 1 && _node->inputs().size() != 1)
            return InferenceEngine::GENERAL_ERROR;

        InferenceEngine::DataConfig cfg;
        cfg.constant = false;
        cfg.inPlace = 0;

        InferenceEngine::SizeVector order;
        auto partialShape = _node->get_output_partial_shape(0);
        if (partialShape.is_dynamic())
            return InferenceEngine::GENERAL_ERROR;

        auto shape = _node->get_output_shape(0);
        for (size_t i = 0; i < shape.size(); i++) {
            order.push_back(i);
        }
        cfg.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, shape, {shape, order});
        layerConfig.outConfs.push_back(cfg);
        layerConfig.inConfs.push_back(cfg);
        conf.push_back(layerConfig);
        return InferenceEngine::OK;
    }

    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& /*config*/, InferenceEngine::ResponseDesc* /*resp*/) noexcept override {
        return InferenceEngine::StatusCode::OK;
    }

    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                        InferenceEngine::ResponseDesc* /*resp*/) noexcept override {
        static bool wasCalled = false;
        if (!wasCalled) {
            std::cout << "Running " + std::string(CUSTOM_RELU_TYPE) + " kernel for the first time (next messages won't be printed)" << std::endl;
            wasCalled = true;
        }
        for (size_t i = 0; i < inputs.size(); i++) {
            InferenceEngine::MemoryBlob::CPtr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(inputs[i]);
            InferenceEngine::MemoryBlob::Ptr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(outputs[i]);
            if (!moutput || !minput) {
                return InferenceEngine::StatusCode::PARAMETER_MISMATCH;
            }
            // locked memory holder should be alive all time while access to its buffer happens
            auto minputHolder = minput->rmap();
            auto moutputHolder = moutput->wmap();

            auto inputData = minputHolder.as<const float*>();
            auto outputData = moutputHolder.as<float*>();
            for (size_t j = 0; j < minput->size(); j++) {
                outputData[j] = inputData[j] < 0 ? 0 : inputData[j];
            }
        }
        return InferenceEngine::StatusCode::OK;
    }

private:
    const std::shared_ptr<ngraph::Node> _node;
};

class CustomReluOp : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info {CUSTOM_RELU_TYPE, 0};
    const ngraph::NodeTypeInfo& get_type_info() const override {
        return type_info;
    }

    CustomReluOp() = default;
    explicit CustomReluOp(const ngraph::Output<ngraph::Node>& arg): Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto input_shape = get_input_partial_shape(0).to_shape();

        ngraph::Shape output_shape(input_shape);
        for (size_t i = 0; i < input_shape.size(); ++i) {
            output_shape[i] = input_shape[i];
        }

        set_output_type(0, get_input_element_type(0), ngraph::PartialShape(output_shape));
    }

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        if (new_args.size() != 1) {
            throw ngraph::ngraph_error("Incorrect number of new arguments");
        }

        return std::make_shared<CustomReluOp>(new_args.at(0));
    }

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        (void)visitor;
        return true;
    }
};

constexpr ngraph::NodeTypeInfo CustomReluOp::type_info;

class InPlaceExtension : public InferenceEngine::IExtension {
public:
    InPlaceExtension() {
        impls[CUSTOM_RELU_TYPE] = [](const std::shared_ptr<ngraph::Node>& node) -> InferenceEngine::ILayerImpl::Ptr {
            return std::make_shared<CustomReLUImpl>(node);
        };
    }

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

    void Unload() noexcept override {}

    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override {
        if (impls.find(node->description()) == impls.end())
            return {};
        return {"CPU"};
    }

    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override {
        if (impls.find(node->description()) == impls.end() || implType != "CPU")
            return nullptr;
        return impls[node->description()](node);
    }

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        static std::map<std::string, ngraph::OpSet> opsets;
        if (opsets.empty()) {
            ngraph::OpSet opset;
            opset.insert<CustomReluOp>();
            opsets["experimental"] = opset;
        }
        return opsets;
    }

private:
    std::map<std::string, std::function<InferenceEngine::ILayerImpl::Ptr(const std::shared_ptr<ngraph::Node>)>> impls;
};
