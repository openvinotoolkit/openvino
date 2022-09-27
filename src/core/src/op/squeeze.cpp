// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/squeeze.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <set>

#include "itt.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/reference/copy.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::Squeeze);

op::Squeeze::Squeeze() : Op() {}

op::Squeeze::Squeeze(const Output<Node>& data, const Output<Node>& axes) : Op({data, axes}) {
    constructor_validate_and_infer_types();
}

op::Squeeze::Squeeze(const Output<Node>& data) : Op({data}) {
    constructor_validate_and_infer_types();
}

void op::Squeeze::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Squeeze_validate_and_infer_types);
    auto data = input_value(0);
    bool data_has_dynamic_rank = data.get_partial_shape().rank().is_dynamic();
    bool data_has_dynamic_shape = data.get_partial_shape().is_dynamic();
    auto data_partial_shape = data.get_partial_shape();

    std::shared_ptr<op::v0::Constant> axes_constant;
    if (get_input_size() == 1) {
        // Handling the case when Squeeze op is created with a single input - data.
        // This way the following code (validation, shape inference) can be used in both cases.
        axes_constant = make_shared<op::v0::Constant>(element::i64, ov::Shape{0}, vector<int64_t>{});
    } else {
        auto axes_node = input_value(1).get_node_shared_ptr();
        auto axes_pshape = get_input_partial_shape(1);
        axes_constant = get_constant_from_source(axes_node);

        NODE_VALIDATION_CHECK(this,
                              axes_pshape.rank().compatible(0) || axes_pshape.rank().compatible(1),
                              "Second input (axes) should not be of rank higher than 1. Got: ",
                              axes_pshape.rank().get_length());
    }

    bool axes_is_empty_constant = (axes_constant && axes_constant->get_data_ptr() != nullptr)
                                      ? axes_constant->cast_vector<int64_t>().empty()
                                      : false;

    if (data_has_dynamic_rank || !axes_constant || !axes_constant->get_data_ptr() ||
        (data_has_dynamic_shape && axes_is_empty_constant)) {
        // If data has a static rank despite being dynamic, it's possible none
        // of the dimensions will be equal to 1. If so, the input shape can be
        // propagated at this point to the output shape.
        if (!data_has_dynamic_rank && axes_is_empty_constant) {
            bool no_squeezable_dimension_present = true;
            uint64_t data_rank = data_partial_shape.rank().get_length();
            for (uint64_t idx = 0; idx < data_rank; ++idx) {
                if (data_partial_shape[idx].compatible(1)) {
                    no_squeezable_dimension_present = false;
                    break;
                }
            }
            if (no_squeezable_dimension_present) {
                set_output_type(0, get_input_element_type(0), data_partial_shape);
                return;
            }
        }

        set_output_type(0, get_input_element_type(0), ov::PartialShape::dynamic());
        return;
    }

    uint64_t data_rank = data_partial_shape.rank().get_length();

    // Get value of axes from Constant
    auto axes = normalize_axes(this->description(), axes_constant->cast_vector<int64_t>(), data_rank);

    // Prepare set of unique axes marked to be removed from input data.
    vector<bool> axes_to_squeeze(data_rank, false);
    if (axes_is_empty_constant) {
        auto data_shape = data.get_shape();
        // Default behaviour is to remove all single dimension axes.
        for (uint64_t idx = 0; idx < data_rank; ++idx) {
            if (data_shape.at(idx) == 1) {
                axes_to_squeeze.at(idx) = true;
            }
        }
    } else {
        set<size_t, greater<size_t>> unique_axes(begin(axes), end(axes));
        for (uint64_t axis : unique_axes) {
            if (!data_has_dynamic_shape) {
                auto data_shape = data.get_shape();
                NODE_VALIDATION_CHECK(this,
                                      (data_shape.at(axis) == 1),
                                      "provided axis value is invalid. Only axes of size 1 may be removed.");
            }
            axes_to_squeeze.at(axis) = true;
        }
    }

    vector<Dimension> output_data_shape;
    for (uint64_t idx = 0; idx < data_rank; ++idx) {
        if (!axes_to_squeeze.at(idx)) {
            output_data_shape.push_back(data_partial_shape[idx]);
        }
    }
    set_output_type(0, get_input_element_type(0), ov::PartialShape(output_data_shape));
}

bool ngraph::op::v0::Squeeze::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Squeeze_visit_attributes);
    return true;
}

shared_ptr<Node> op::Squeeze::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Squeeze_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 1) {
        return make_shared<Squeeze>(new_args.at(0));
    } else if (new_args.size() == 2) {
        return make_shared<Squeeze>(new_args.at(0), new_args.at(1));
    } else {
        throw ngraph_error("Incorrect number of new arguments");
    }
}

namespace squeeze {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& arg1, const HostTensorPtr& out) {
    const auto data_rank = arg0->get_partial_shape().rank().get_length();
    const auto axes_num = shape_size(arg1->get_shape());

    auto out_shape = arg0->get_shape();
    if (axes_num == 0) {
        out_shape.erase(remove(out_shape.begin(), out_shape.end(), 1), out_shape.end());
    } else {
        auto norm_axes =
            normalize_axes("",
                           std::vector<int64_t>(arg1->get_data_ptr<ET>(), arg1->get_data_ptr<ET>() + axes_num),
                           data_rank);
        set<size_t, greater<size_t>> ordered_axes(norm_axes.begin(), norm_axes.end());

        for (const auto& axis : ordered_axes) {
            if (out_shape[axis] != 1) {
                throw ngraph_error("Squeeze dimension is not equal to 1");
            }
            out_shape.erase(out_shape.begin() + axis);
        }
    }
    out->set_shape(out_shape);

    runtime::reference::copy(arg0->get_data_ptr<char>(),
                             out->get_data_ptr<char>(),
                             shape_size(out_shape) * out->get_element_type().size());
    return true;
}

bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out) {
    auto out_shape = arg0->get_shape();

    out_shape.erase(remove(out_shape.begin(), out_shape.end(), 1), out_shape.end());

    out->set_shape(out_shape);

    runtime::reference::copy(arg0->get_data_ptr<char>(),
                             out->get_data_ptr<char>(),
                             shape_size(out_shape) * out->get_element_type().size());
    return true;
}

bool evaluate_squeeze(const HostTensorPtr& arg0, const HostTensorPtr& arg1, const HostTensorPtr& out) {
    auto element_type = arg1->get_element_type();

    bool rc = true;
    switch (element_type) {
        NGRAPH_TYPE_CASE(evaluate_squeeze, i8, arg0, arg1, out);
        NGRAPH_TYPE_CASE(evaluate_squeeze, i16, arg0, arg1, out);
        NGRAPH_TYPE_CASE(evaluate_squeeze, i32, arg0, arg1, out);
        NGRAPH_TYPE_CASE(evaluate_squeeze, i64, arg0, arg1, out);
        NGRAPH_TYPE_CASE(evaluate_squeeze, u8, arg0, arg1, out);
        NGRAPH_TYPE_CASE(evaluate_squeeze, u16, arg0, arg1, out);
        NGRAPH_TYPE_CASE(evaluate_squeeze, u32, arg0, arg1, out);
        NGRAPH_TYPE_CASE(evaluate_squeeze, u64, arg0, arg1, out);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool evaluate_squeeze(const HostTensorPtr& arg0, const HostTensorPtr& out) {
    return evaluate(arg0, out);
}
}  // namespace
}  // namespace squeeze

bool op::v0::Squeeze::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Squeeze_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, inputs.size()));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));

    if (inputs.size() == 1) {
        return squeeze::evaluate_squeeze(inputs[0], outputs[0]);
    }

    return squeeze::evaluate_squeeze(inputs[0], inputs[1], outputs[0]);
}

bool op::v0::Squeeze::has_evaluate() const {
    OV_OP_SCOPE(v0_Squeeze_has_evaluate);

    if (get_input_size() == 2) {
        switch (get_input_element_type(1)) {
        case ngraph::element::i8:
        case ngraph::element::i16:
        case ngraph::element::i32:
        case ngraph::element::i64:
        case ngraph::element::u8:
        case ngraph::element::u16:
        case ngraph::element::u32:
        case ngraph::element::u64:
            return true;
        default:
            break;
        }
        return false;
    } else if (get_input_size() == 1) {
        return true;
    } else {
        return false;
    }
}

bool op::v0::Squeeze::evaluate_lower(const HostTensorVector& output_values) const {
    OV_OP_SCOPE(v0_Squeeze_evaluate_lower);
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));

    if (inputs().size() > 1 && !input_value(1).get_tensor().has_and_set_bound())
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v0::Squeeze::evaluate_upper(const HostTensorVector& output_values) const {
    OV_OP_SCOPE(v0_Squeeze_evaluate_upper);
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));

    if (inputs().size() > 1 && !input_value(1).get_tensor().has_and_set_bound())
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool op::v0::Squeeze::evaluate_label(TensorLabelVector& output_labels) const {
    if (get_input_size() > 1 && !get_input_tensor(1).has_and_set_bound())
        return false;
    return default_label_evaluator(this, output_labels);
}

bool op::v0::Squeeze::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    OV_OP_SCOPE(v0_Squeeze_constant_fold);
    if (get_output_partial_shape(0).is_dynamic() || is_const_fold_disabled()) {
        return false;
    }

    const auto& shape = get_output_shape(0);

    if (auto data_const = std::dynamic_pointer_cast<op::v0::Constant>(inputs_values[0].get_node_shared_ptr())) {
        output_values[0] = std::make_shared<op::v0::Constant>(*data_const, shape);
        return true;
    }
    return false;
}

bool op::v0::Squeeze::is_dynamic() const {
    return get_output_partial_shape(0).is_dynamic();
}
