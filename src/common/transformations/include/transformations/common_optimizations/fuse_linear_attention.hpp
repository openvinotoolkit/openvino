#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API LinearAttentionFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses linear attention sub-graph into internal LinearAttention operation.
 */
class ov::pass::LinearAttentionFusion : public ov::pass::MatcherPass {
public:
	OPENVINO_MATCHER_PASS_RTTI("LinearAttentionFusion");
	LinearAttentionFusion();
};
