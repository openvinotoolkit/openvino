// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <algorithm>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/validation_util.hpp>


namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertReduceToPooling);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertReduceToPooling: public ngraph::pass::GraphRewrite {
public:
    ConvertReduceToPooling() : GraphRewrite() {
        convert_reduce_to_pooling<ngraph::opset1::ReduceMean>();
        convert_reduce_to_pooling<ngraph::opset1::ReduceMax>();
        convert_reduce_to_pooling<ngraph::opset1::ReduceSum>();
    }

private:
    template <class T>
    void convert_reduce_to_pooling();
};

template <class T>
void ngraph::pass::ConvertReduceToPooling::convert_reduce_to_pooling() {
    {
        static_assert(std::is_same<T, ngraph::opset1::ReduceMean>() ||
                      std::is_same<T, ngraph::opset1::ReduceMax>()  ||
                      std::is_same<T, ngraph::opset1::ReduceSum>(),
                      "This callback works only with ngraph::opset1::ReduceMean/Max/Sum");

        auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto axes = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
        auto reduce = std::make_shared<T>(data, axes);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto reduce = std::dynamic_pointer_cast<T>(m.get_match_root());
            if (!reduce) {
                return false;
            }

            auto input = reduce->input(0).get_source_output().get_node_shared_ptr();

            auto axes_node = reduce->input(1).get_source_output().get_node_shared_ptr();
            if (!std::dynamic_pointer_cast<ngraph::opset1::Constant>(axes_node)) {
                return false;
            }

            auto axes_vector = std::dynamic_pointer_cast<ngraph::opset1::Constant>(axes_node)->template get_vector<int64_t>();
            auto input_shape = reduce->input(0).get_shape();
            auto input_rank = input_shape.size();
            // Transform negative axes into non-negative ones
            for (size_t i = 0; i < axes_vector.size(); ++i) {
                if (axes_vector[i] < 0) {
                    axes_vector[i] += input_rank;
                }
            }
            std::sort(axes_vector.begin(), axes_vector.end());

            // If axes are empty we just remove Reduction operation
            if (axes_vector.empty()) {
                replace_node(reduce, input);
                return true;
            }

            // Check that axes are consecutive otherwise this transformation is not applicable
            for (size_t i = 1; i < axes_vector.size(); ++i) {
                if (axes_vector[i] - axes_vector[i-1] != 1) {
                    return false;
                }
            }

            // As this transformation requires static input shape we should guaranty it
            if (reduce->input(0).get_partial_shape().is_dynamic()) {
                return false;
            }

            // Check either reduction applies to spatial dimensions or not
            bool spatial_dims_reduction(true);
            size_t reduction_dims_count = 1;
            for (auto& axis : axes_vector) {
                reduction_dims_count *= input_shape[axis];
                if (axis <= 1) {
                    spatial_dims_reduction = false;
                }
            }

            /*
             * Prepare default attributes for Pooling operation
             *      pads_begin/pads_end - should be zeros as we don't need any padding
             *      stride - should be filled with ones
             *      kernel - depends on Reduction operation axes
             *
             * Also here we decide should we use Reshapes before and after Pooling
             *      shape_begin - if not empty indicates that we need a Reshape before Pooling
             *      shape_end   - if not empty indicates that we need a Reshape after Pooling
             */

            Strides strides;
            Shape pads_begin, pads_end, kernel, shape_begin, shape_end;

            if (!spatial_dims_reduction || input_shape.size() != 4) {
                // In case if reduction applies not to spatial dimensions
                // we have to fit it into 4D Pooling
                size_t dims_prod = 1, dims_begin = 1, dims_end = 1;
                for (size_t i = 0; i < input_shape.size(); ++i) {
                    if (i < *axes_vector.begin()) {
                        dims_begin *= input_shape[i];
                    } else if (i >= axes_vector.front() && i <= axes_vector.back()) {
                        dims_prod *= input_shape[i];
                    } else {
                        dims_end *= input_shape[i];
                    }
                }
                // The batch dimenstion is repositioned in the shape
                // only in case of batch dimension reduction
                shape_begin.assign({dims_begin, 1, dims_prod, dims_end});
                shape_end = reduce->output(0).get_shape();
                strides.assign({1, 1});
                pads_begin.assign({0, 0});
                pads_end.assign({0, 0});
                kernel.assign({dims_prod, 1});
            } else {
                for (size_t i = 0; i < input_shape.size() - 2; ++i) {
                    strides.push_back(1);
                    pads_begin.push_back(0);
                    pads_end.push_back(0);
                    kernel.push_back(1);
                }
                for (auto& axis : axes_vector) {
                    kernel[axis-2] = input_shape[axis];
                }
                if (!reduce->get_keep_dims()) {
                    shape_end = reduce->output(0).get_shape();
                }
            }

            /*
             *  ReduceMean => AvgPool
             *                AvgPool->Reshape (in case if keep_dims=False)
             *                Reshape->AvgPool->Reshape (in case if axes doesn't match spatial dims)

             *  ReduceMax  => MaxPool
             *                MaxPool->Reshape (in case if keep_dims=False)
             *                Reshape->MaxPool->Reshape (in case if axes doesn't match spatial dims)
             *
             *  ReduceSum  => AvgPool->Multiply
             *                AvgPool->Multiply->Reshape (in case if keep_dims=False)
             *                Reshape->AvgPool->Multiply->Reshape (in case if axes doesn't match spatial dims)
             *
             *  Note: some of reshape nodes can be optimized if they do nothing.
             */

            if (!shape_begin.empty() && shape_begin != input->output(0).get_shape()) {
                input = std::make_shared<ngraph::opset1::Reshape>(input, opset1::Constant::create(element::i64, Shape{shape_begin.size()}, shape_begin), true);
                input->set_friendly_name(reduce->get_friendly_name() + "/reshape_begin");
            }

            if (std::is_same<T, ngraph::opset1::ReduceMean>()) {
                input = std::make_shared<ngraph::opset1::AvgPool>(input,
                                                                  strides,
                                                                  pads_begin,
                                                                  pads_end,
                                                                  kernel,
                                                                  true,
                                                                  op::RoundingType::FLOOR);

                input->set_friendly_name(reduce->get_friendly_name() + "/pool");
            } else if (std::is_same<T, ngraph::opset1::ReduceMax>()) {
                input = std::make_shared<ngraph::opset1::MaxPool>(input,
                                                                  strides,
                                                                  pads_begin,
                                                                  pads_end,
                                                                  kernel,
                                                                  op::RoundingType::FLOOR);

                input->set_friendly_name(reduce->get_friendly_name() + "/pool");
            } else if (std::is_same<T, ngraph::opset1::ReduceSum>()) {
                input = std::make_shared<ngraph::opset1::AvgPool>(input,
                                                                  strides,
                                                                  pads_begin,
                                                                  pads_end,
                                                                  kernel,
                                                                  true,
                                                                  op::RoundingType::FLOOR);

                input->set_friendly_name(reduce->get_friendly_name() + "/pool");

                input = std::make_shared<ngraph::opset1::Multiply>(input,
                        opset1::Constant::create(reduce->input(0).get_element_type(), Shape{1}, {reduction_dims_count}));
                input->set_friendly_name(reduce->get_friendly_name() + "/mul");
            } else {
                return false;
            }

            if (!shape_end.empty() && shape_end != input->output(0).get_shape()) {
                input = std::make_shared<ngraph::opset1::Reshape>(input, opset1::Constant::create(element::i64, Shape{shape_end.size()}, shape_end), true);
            }
            input->set_friendly_name(reduce->get_friendly_name());

            replace_node(reduce, input);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(reduce, "ConvertReduceToPooling");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
}
