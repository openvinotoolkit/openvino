#pragma once

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "ngraph/opsets/opset3.hpp"
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

    class AddDecomposition;
    class SubDecomposition;
    class MulDecomposition;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::AddDecomposition: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

class ngraph::pass::SubDecomposition: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

class ngraph::pass::MulDecomposition: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};