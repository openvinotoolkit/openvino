// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/scatter_update.hpp"

#include "itt.hpp"
#include "ngraph/runtime/reference/scatter_update.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::v3::ScatterUpdate::ScatterUpdate(const Output<Node>& data,
                                     const Output<Node>& indices,
                                     const Output<Node>& updates,
                                     const Output<Node>& axis)
    : util::ScatterBase(data, indices, updates, axis) {}

shared_ptr<Node> op::v3::ScatterUpdate::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_ScatterUpdate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v3::ScatterUpdate>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

namespace scatter_update {
namespace {
template <element::Type_t ET>
std::vector<int64_t> get_indices(const HostTensorPtr& in) {
    auto data_ptr = in->get_data_ptr<ET>();
    return std::vector<int64_t>(data_ptr, data_ptr + in->get_element_count());
}
}  // namespace
}  // namespace scatter_update

#define GET_INDICES(a, ...)                                                                   \
    case element::Type_t::a: {                                                                \
        OV_OP_SCOPE(OV_PP_CAT3(get_scatter_update_indices, _, a));                            \
        indices_casted_vector = scatter_update::get_indices<element::Type_t::a>(__VA_ARGS__); \
    } break;

bool op::v3::ScatterUpdate::evaluate_scatter_update(const HostTensorVector& outputs,
                                                    const HostTensorVector& inputs) const {
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    const auto& axis = inputs[3];
    const auto& out = outputs[0];

    const auto elem_size = data->get_element_type().size();
    out->set_shape(data->get_shape());

    NGRAPH_CHECK(axis->get_element_type().is_integral_number(), "axis element type is not integral data type");

    int64_t axis_val = host_tensor_2_vector<int64_t>(axis)[0];
    if (axis_val < 0) {
        axis_val = ngraph::normalize_axis(this, axis_val, static_cast<int64_t>(data->get_shape().size()));
    }

    std::vector<int64_t> indices_casted_vector;
    switch (indices->get_element_type()) {
        GET_INDICES(i8, indices);
        GET_INDICES(i16, indices);
        GET_INDICES(i32, indices);
        GET_INDICES(i64, indices);
        GET_INDICES(u8, indices);
        GET_INDICES(u16, indices);
        GET_INDICES(u32, indices);
        GET_INDICES(u64, indices);
    default:
        return false;
    }

    ngraph::runtime::reference::scatter_update(data->get_data_ptr<char>(),
                                               indices_casted_vector.data(),
                                               updates->get_data_ptr<char>(),
                                               axis_val,
                                               out->get_data_ptr<char>(),
                                               elem_size,
                                               data->get_shape(),
                                               indices->get_shape(),
                                               updates->get_shape());

    return true;
}

bool op::v3::ScatterUpdate::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v3_ScatterUpdate_evaluate);
    return evaluate_scatter_update(outputs, inputs);
}

bool op::v3::ScatterUpdate::evaluate_lower(const HostTensorVector& outputs) const {
    OV_OP_SCOPE(v3_ScatterUpdate_evaluate_lower);
    return get_input_tensor(1).has_and_set_bound() && get_input_tensor(3).has_and_set_bound() &&
           default_lower_bound_evaluator(this, outputs);
}

bool op::v3::ScatterUpdate::evaluate_upper(const HostTensorVector& outputs) const {
    OV_OP_SCOPE(v3_ScatterUpdate_evaluate_upper);
    return get_input_tensor(1).has_and_set_bound() && get_input_tensor(3).has_and_set_bound() &&
           default_upper_bound_evaluator(this, outputs);
}

bool op::v3::ScatterUpdate::has_evaluate() const {
    OV_OP_SCOPE(v3_ScatterUpdate_has_evaluate);

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
}

namespace {
bool scatter_label_evaluator(const Node* node, TensorLabelVector& output_labels) {
    const auto& input_values = node->input_values();

    constexpr auto data_in_idx = 0;
    constexpr auto updates_in_idx = 2;
    std::vector<size_t> data_labels = input_values[data_in_idx].get_tensor().get_value_label();
    std::vector<size_t> updates_labels = input_values[updates_in_idx].get_tensor().get_value_label();

    if (ov::has_no_labels(data_labels) && ov::has_no_labels(updates_labels)) {
        return false;
    }

    constexpr auto element_type = (sizeof(size_t) == 8) ? element::u64 : element::u32;
    std::vector<ov::runtime::Tensor> input_tensors;
    input_tensors.reserve(input_values.size());

    auto make_input_label = [&](const Output<Node>& input, TensorLabel& labels) {
        input_tensors.emplace_back(element_type, input.get_shape());
        labels.resize(shape_size(input.get_shape()));
        memcpy(input_tensors.back().data(), labels.data(), input_tensors.back().get_byte_size());
    };

    for (size_t i = 0; i < input_values.size(); ++i) {
        const auto& input = input_values[i];
        if (i == data_in_idx) {
            make_input_label(input, data_labels);
        } else if (i == updates_in_idx) {
            make_input_label(input, updates_labels);
        } else {
            const auto host_tensor_ptr = input.get_tensor().get_lower_value();
            input_tensors.emplace_back(host_tensor_ptr->get_element_type(),
                                       host_tensor_ptr->get_shape(),
                                       host_tensor_ptr->get_data_ptr());
        }
    }

    ov::TensorVector output_tensors{ov::Tensor(element_type, node->get_output_shape(0))};
    if (node->evaluate(output_tensors, input_tensors)) {
        size_t* ptr = static_cast<size_t*>(output_tensors[0].data(element_type));
        output_labels[0] = std::vector<size_t>(ptr, ptr + output_tensors[0].get_size());
        return true;
    }
    return false;
}
}  // namespace

bool op::v3::ScatterUpdate::evaluate_label(TensorLabelVector& output_labels) const {
    OV_OP_SCOPE(v3_ScatterUpdate_evaluate_label);
    if (get_input_partial_shape(0).is_static() && get_input_partial_shape(2).is_static() &&
        get_input_tensor(1).has_and_set_bound() && get_input_tensor(3).has_and_set_bound()) {
        return scatter_label_evaluator(this, output_labels);
    }
    return false;
}
