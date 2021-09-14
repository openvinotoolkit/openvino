// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tensorflow_frontend/frontend.hpp>

//#include <ngraph/pass/transpose_sinking.h>
#include <ngraph/pass/constant_folding.hpp>

#include "ngraph_builder.h"
#include "tf_framework_node.hpp"

using namespace google;

using namespace ngraph::frontend;

using ::tensorflow::GraphDef;

/// \brief Check if FrontEndTensorflow can recognize model from given parts
bool FrontEndTensorflow::supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    // TODO: Support TensorFlow 2 SavedModel format
    if (variants.empty() || variants.size() > 2)
        return false;

    // Validating first path, it must contain a model
    if (ov::is_type<VariantWrapper<std::string>>(variants[0])) {
        std::string suffix = ".pb";
        std::string model_path = ov::as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
        if (tf::endsWith(model_path, suffix)) {
            return true;
        }
    }
    return false;
}

InputModel::Ptr FrontEndTensorflow::load_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    if (variants.size() == 1) {
        // The case when folder with __model__ and weight files is provided or .pdmodel file
        if (ov::is_type<VariantWrapper<std::string>>(variants[0])) {
            std::string m_path = ov::as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
            return std::make_shared<InputModelTensorflow>(m_path);
        }
    }
    return nullptr;
}

std::shared_ptr<ngraph::Function> FrontEndTensorflow::convert(InputModel::Ptr model) const {
    auto model_tf = std::dynamic_pointer_cast<ngraph::frontend::InputModelTensorflow>(model);
    std::cout << "[ INFO ] FrontEndTensorflow::convert invoked\n";

    std::shared_ptr<ngraph::Function> f;
    ::tensorflow::ngraph_bridge::Builder::TranslateGraph(model_tf, {}, "here_should_be_a_graph_name", true, false, f);
    std::cout << "[ STATUS ] TranslateGraph was called successfuly.\n";
    std::cout << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << " nodes." << std::endl;

    normalize(f);

    // TODO: check that ngraph function doesn't contain operations which are not in the opset

    return f;
}

std::shared_ptr<ngraph::Function> FrontEndTensorflow::convert_partially(InputModel::Ptr model) const {
    auto model_tf = std::dynamic_pointer_cast<ngraph::frontend::InputModelTensorflow>(model);
    std::cout << "[ INFO ] FrontEndTensorflow::convert_partially invoked\n";

    std::shared_ptr<ngraph::Function> f;
    ::tensorflow::ngraph_bridge::Builder::TranslateGraph(model_tf, {}, "here_should_be_a_graph_name", false, false, f);
    std::cout << "[ STATUS ] TranslateGraph was called successfuly.\n";
    std::cout << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << " nodes." << std::endl;

    normalize(f);
    return f;
}

std::shared_ptr<ngraph::Function> FrontEndTensorflow::decode(InputModel::Ptr model) const {
    auto model_tf = std::dynamic_pointer_cast<ngraph::frontend::InputModelTensorflow>(model);
    std::cout << "[ INFO ] FrontEndTensorflow::decode invoked\n";

    std::shared_ptr<ngraph::Function> f;
    ::tensorflow::ngraph_bridge::Builder::TranslateGraph(model_tf, {}, "here_should_be_a_graph_name", false, true, f);
    std::cout << "[ STATUS ] TranslateGraphFWNode was called successfuly.\n";
    std::cout << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << " nodes." << std::endl;
    return f;
}

void FrontEndTensorflow::convert(std::shared_ptr<ngraph::Function> partiallyConverted) const {
    for (const auto& node : partiallyConverted->get_ordered_ops()) {
        if (ov::is_type<TFFrameworkNode>(node)) {
            ::tensorflow::ngraph_bridge::Builder::TranslateFWNode(std::dynamic_pointer_cast<TFFrameworkNode>(node));
        }
    }
    for (auto result : partiallyConverted->get_results()) {
        result->validate_and_infer_types();
    }

    normalize(partiallyConverted);
}

void FrontEndTensorflow::normalize(std::shared_ptr<ngraph::Function> function) const {
    std::cout << "[ STATUS ] Running Transpose Sinking transformation\n";

    ngraph::pass::Manager manager;
    // manager.register_pass<ngraph::pass::TransposeSinking>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.run_passes(function);

    std::cout << "[ INFO ] Resulting nGraph function contains " << function->get_ops().size() << " nodes." << std::endl;
}
