#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief TODO
 */
class Reshape1DConvolution : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  Reshape1DConvolution();
};

} // namespacec GNAPluginNS