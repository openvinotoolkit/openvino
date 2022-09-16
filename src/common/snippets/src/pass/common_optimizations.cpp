// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/common_optimizations.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "transformations/utils/utils.hpp"
#include "transformations/op_conversions/fq_decomposition.hpp"
#include "snippets/pass/transform_convert.hpp"
#include "snippets/pass/insert_convert.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::CommonOptimizations, "Snippets::CommonOptimizations", 0);

namespace ngraph {
namespace snippets {
namespace pass {

CommonOptimizations::CommonOptimizations() {
    auto wrapper = ngraph::pattern::wrap_type<ngraph::snippets::op::Subgraph>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::CommonOptimizations");

        auto subgraph = ngraph::as_type_ptr<ngraph::snippets::op::Subgraph>(m.get_match_root());
        if (transformation_callback(subgraph)) {
            return false;
        }

        auto body = subgraph->get_body();

        // Firsly we should transform all original Converts inside body to ConvertTruncation to save original behavior.
        // Then if Subgraph contains FakeQuantize we enable low precision specific transformation. Before we should decompose FakeQuantize into simple operations.
        // After FQ decomposition we should transform new Converts to ConvertSaturation to save saturation behavior.
        // Also we have to insert reverse converts after ConvertSaturation (after FQ decompoisition) to return FP32 calculation inside body
        // TODO: We disable forse rounding inside Subgraph because ConvertSaturation after decomposition correctly round values (half to even).
        //       But original Convert from specification uses truncation rounding. So when plugin supports all FQ as Subgraphs (limitations on inputs)
        //       we should remove this force rounding
        ngraph::pass::Manager manager;
        manager.set_per_pass_validation(false);
        manager.register_pass<ngraph::snippets::pass::TransformConvertToConvertTruncation>();
        if (subgraph->is_low_precision()) {
            manager.register_pass<ngraph::pass::FakeQuantizeDecomposition>(false, false);
            manager.register_pass<ngraph::pass::ConstantFolding>();
            manager.register_pass<ngraph::pass::Validate>();
            manager.register_pass<ngraph::snippets::pass::TransformConvertToConvertSaturation>();
            manager.register_pass<ngraph::snippets::pass::InsertReverseConvert>();
            manager.register_pass<ngraph::pass::Validate>();
        }
        manager.run_passes(body);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(wrapper, "snippets::pass::CommonOptimizations");
    this->register_matcher(m, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ngraph
