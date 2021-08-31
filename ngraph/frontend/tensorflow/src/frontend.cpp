// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tensorflow_frontend/frontend.hpp>

//#include <ngraph/pass/transpose_sinking.h>
#include <ngraph/pass/constant_folding.hpp>

#include "ngraph_builder.h"

using namespace google;

using namespace ngraph::frontend;

using ::tensorflow::GraphDef;

std::shared_ptr<ngraph::Function> ngraph::frontend::FrontEndTensorflow::convert(InputModel::Ptr model) const {
    try {
        auto model_tf = std::dynamic_pointer_cast<ngraph::frontend::InputModelTensorflow>(model);
        std::cout << "[ INFO ] FrontEndTensorflow::convert invoked\n";

        std::shared_ptr<ngraph::Function> f;
        ::tensorflow::ngraph_bridge::Builder::TranslateGraph(model_tf->partialShapes,
                                                             model_tf->input_shapes,
                                                             {},
                                                             *model_tf->graph_impl,
                                                             "here_should_be_a_graph_name",
                                                             f);
        std::cout << "[ STATUS ] TranslateGraph was called successfuly.\n";
        std::cout << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << " nodes." << std::endl;
        std::cout << "[ STATUS ] Running Transpose Sinking transformation\n";

        ngraph::pass::Manager manager;
        // manager.register_pass<ngraph::pass::TransposeSinking>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(f);

        std::cout << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << " nodes." << std::endl;
        return f;
    } catch (::tensorflow::Status status) {
        std::cerr << "[ ERROR ] Exception happens during TF model conversion: " << status << "\n";
        throw;
    }
}
