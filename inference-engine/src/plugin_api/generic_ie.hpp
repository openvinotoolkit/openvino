// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>

#include <ie_iextension.h>
#include <ie_parameter.hpp>
#include <ie_precision.hpp>
#include "ngraph/op/op.hpp"
#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace op {

/**
 * This generic operation is necessary for legacy scenario.
 * If user has old shape inference extensions, this node allow to use it for shape inference
 */
class INFERENCE_ENGINE_API_CLASS(GenericIE) : public Op {
public:
    struct PortIE {
        InferenceEngine::Precision precision;
        std::vector<size_t> dims;
    };

    class DisableReshape {
    public:
        explicit DisableReshape(std::vector<std::shared_ptr<ngraph::Node>>& ops) {
            for (auto& op : ops) {
                addOp(op);
            }
        }
        explicit DisableReshape(const std::shared_ptr<const ngraph::Function>& graph) {
            if (!graph)
                return;

            for (auto& op : graph->get_ops()) {
                addOp(op);
            }
        }

        ~DisableReshape() {
            for (auto& generic : genericOps) {
                generic->doReshape(true);
            }
        }

    private:
        std::vector<std::shared_ptr<ngraph::op::GenericIE>> genericOps;

        void addOp(std::shared_ptr<ngraph::Node>& op) {
            if (auto generic = std::dynamic_pointer_cast<GenericIE>(op)) {
                generic->doReshape(false);
                genericOps.emplace_back(generic);
            }
            if (auto ti_node = std::dynamic_pointer_cast<ngraph::op::TensorIterator>(op)) {
                auto results = ti_node->get_body()->get_results();
                auto params = ti_node->get_body()->get_parameters();
                ngraph::NodeVector nResults, nParams;
                for (const auto& res : results)
                    nResults.emplace_back(res);
                for (const auto& param : params)
                    nParams.emplace_back(param);
                ngraph::traverse_nodes(nResults, [&](std::shared_ptr<ngraph::Node> node) {
                    if (auto genNode = std::dynamic_pointer_cast<ngraph::op::GenericIE>(node)) {
                        genNode->doReshape(false);
                        genericOps.emplace_back(genNode);
                    }
                }, true, nParams);
            }
        }
    };

    static constexpr NodeTypeInfo type_info{"GenericIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    /**
     * @brief constructor of generic node
     *
     * @param inputs vector of inputs
     * @param params map of parameters (std::string, Blob::Ptr, Blob::CPtr)
     * @param type string with original layer type
     * @param outputs information about output ports from IR
     */
    GenericIE(const NodeVector& inputs,
              const std::map<std::string, InferenceEngine::Parameter>& params,
              const std::string type,
              const std::vector<PortIE>& outputs);

    GenericIE(const OutputVector& inputs,
              const std::map<std::string, InferenceEngine::Parameter>& params,
              const std::string type,
              const std::vector<PortIE>& outputs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    static void addExtension(std::shared_ptr<const ngraph::Lambda> func, const InferenceEngine::IShapeInferExtensionPtr& ext);
    static std::vector<InferenceEngine::IShapeInferExtensionPtr> getExtensions(std::shared_ptr<const ngraph::Function> func);

    const std::string& getType() const {
        return type;
    }

    const std::map<std::string, InferenceEngine::Parameter>& getParameters() const {
        return params;
    }

private:
    void doReshape(bool flag) {
        reshape = flag;
    }

    std::vector<InferenceEngine::IShapeInferExtensionPtr> extensions;
    bool reshape = true;
    std::map<std::string, InferenceEngine::Parameter> params;
    std::vector<PortIE> outputs;
    std::string type;
    int initialized;

    void addExtension(const InferenceEngine::IShapeInferExtensionPtr& ext);
    std::vector<InferenceEngine::IShapeInferExtensionPtr> getExtensions();
};

}  // namespace op
}  // namespace ngraph

