
#pragma once

#include <ngraph/pass/pass.hpp>
#include <ngraph/util.hpp>

namespace ngraph {
namespace pass {

class NGRAPH_API CoreTransposeSinking : public ngraph::pass::FunctionPass {
 public:
    CoreTransposeSinking() {
    set_property(ngraph::pass::PassProperty::REQUIRE_STATIC_SHAPE, true);
  }
  bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};

}  // namespace pass
}  // namespace ngraph
