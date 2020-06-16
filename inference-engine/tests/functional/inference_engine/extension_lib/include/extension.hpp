// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <ie_api.h>
#include <ngraph/ngraph.hpp>
#include <memory>
#include <vector>
#include <string>
#include <map>

IE_SUPPRESS_DEPRECATED_START

class ExtensionTestOp : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"Test", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    ExtensionTestOp() = default;
    explicit ExtensionTestOp(const ngraph::Output<ngraph::Node>& arg): Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto input_shape = get_input_partial_shape(0).to_shape();

        ngraph::Shape output_shape(input_shape);
        for (int i = 0; i < input_shape.size(); ++i) {
            output_shape[i] = input_shape[i];
        }

        set_output_type(0, get_input_element_type(0), ngraph::PartialShape(output_shape));
    }

    std::shared_ptr<ngraph::Node> copy_with_new_args(const ngraph::NodeVector& new_args) const override {
        if (new_args.size() != 1) {
            throw ngraph::ngraph_error("Incorrect number of new arguments");
        }

        return std::make_shared<ExtensionTestOp>(new_args.at(0));
    }

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        return true;
    }
};

class TestExtension : public InferenceEngine::IExtension {
public:
    TestExtension() = default;
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override;
    void Unload() noexcept override {}
    void Release() noexcept override {
        delete this;
    }
    InferenceEngine::StatusCode getFactoryFor(InferenceEngine::ILayerImplFactory*& factory,
                                              const InferenceEngine::CNNLayer* cnnLayer,
                                              InferenceEngine::ResponseDesc* resp) noexcept override {
        if (cnnLayer == nullptr || cnnLayer->type != "test")
            return InferenceEngine::GENERAL_ERROR;
        return InferenceEngine::OK;
    }

    /**
     * @brief Fills passed array with types of layers which kernel implementations are included in the extension
     *
     * @param types Array to store the layer types
     * @param size Size of the layer types array
     * @param resp Response descriptor
     * @return Status code
     */
    InferenceEngine::StatusCode getPrimitiveTypes(char**& types, unsigned int& size, InferenceEngine::ResponseDesc* resp) noexcept override {
        size = 1;
        return InferenceEngine::OK;
    }

    InferenceEngine::StatusCode getShapeInferTypes(char**& types, unsigned int& size, InferenceEngine::ResponseDesc* resp) noexcept override {
        size = 1;
        return InferenceEngine::OK;
    }

    InferenceEngine::StatusCode getShapeInferImpl(InferenceEngine::IShapeInferImpl::Ptr& impl, const char* type,
                                                  InferenceEngine::ResponseDesc* resp) noexcept override {
        std::string type_str = type;
        if (type_str != "test")
            return InferenceEngine::GENERAL_ERROR;
        return InferenceEngine::OK;
    }

    /**
     * @brief Returns operation sets
     * This method throws an exception if it was not implemented
     * @return map of opset name to opset
     */
    std::map<std::string, ngraph::OpSet> getOpSets() override;

    /**
     * @brief Returns vector of implementation types
     * @param node shared pointer to nGraph op
     * @return vector of strings
     */
    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override;

    /**
     * @brief Returns implementation for specific nGraph op
     * @param node shared pointer to nGraph op
     * @param implType implementation type
     * @return shared pointer to implementation
     */
    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override;
};

IE_SUPPRESS_DEPRECATED_END
