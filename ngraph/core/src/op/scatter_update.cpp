// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/scatter_update.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/scatter_update.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_RTTI_DEFINITION(op::v3::ScatterUpdate, "ScatterUpdate", 3, util::ScatterBase);

op::v3::ScatterUpdate::ScatterUpdate(const Output<Node>& data,
                                     const Output<Node>& indices,
                                     const Output<Node>& updates,
                                     const Output<Node>& axis)
    : util::ScatterBase(data, indices, updates, axis) {}

shared_ptr<Node> op::v3::ScatterUpdate::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v3_ScatterUpdate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v3::ScatterUpdate>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

namespace scatter_update {
template <element::Type_t ET>
std::vector<int64_t> get_indices(const HostTensorPtr& in) {
    auto data_ptr = in->get_data_ptr<ET>();
    return std::vector<int64_t>(data_ptr, data_ptr + in->get_element_count());
}

#define GET_INDICES(a, ...)                                                                   \
    case element::Type_t::a: {                                                                \
        NGRAPH_OP_SCOPE(OV_PP_CAT3(get_scatter_update_indices, _, a));                        \
        indices_casted_vector = scatter_update::get_indices<element::Type_t::a>(__VA_ARGS__); \
    } break;

template <element::Type_t ET>
bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
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
        // axis_val = ngraph::normalize_axis(this, axis_val, static_cast<int64_t>(data->get_shape().size()));
        axis_val = axis_val + data->get_shape().size();
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

    runtime::reference::scatter_update<T>(data->get_data_ptr<ET>(),
                                          indices_casted_vector.data(),
                                          updates->get_data_ptr<ET>(),
                                          axis_val,
                                          out->get_data_ptr<ET>(),
                                          elem_size,
                                          data->get_shape(),
                                          indices->get_shape(),
                                          updates->get_shape());

    return true;
}

bool evaluate_scatter_update(const HostTensorVector& outputs, const HostTensorVector& inputs) {
    bool rc = true;
    const auto out = outputs[0];
    switch (out->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_scatter_update, f16, outputs, inputs);
        NGRAPH_TYPE_CASE(evaluate_scatter_update, f32, outputs, inputs);
        NGRAPH_TYPE_CASE(evaluate_scatter_update, bf16, outputs, inputs);
        NGRAPH_TYPE_CASE(evaluate_scatter_update, i8, outputs, inputs);
        NGRAPH_TYPE_CASE(evaluate_scatter_update, i16, outputs, inputs);
        NGRAPH_TYPE_CASE(evaluate_scatter_update, i32, outputs, inputs);
        NGRAPH_TYPE_CASE(evaluate_scatter_update, i64, outputs, inputs);
        NGRAPH_TYPE_CASE(evaluate_scatter_update, u8, outputs, inputs);
        NGRAPH_TYPE_CASE(evaluate_scatter_update, u16, outputs, inputs);
        NGRAPH_TYPE_CASE(evaluate_scatter_update, u32, outputs, inputs);
        NGRAPH_TYPE_CASE(evaluate_scatter_update, u64, outputs, inputs);
    default:
        rc = false;
        break;
    }
    return rc;
}

}  // namespace scatter_update

bool op::v3::ScatterUpdate::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    NGRAPH_OP_SCOPE(v3_ScatterUpdate_evaluate);
    NGRAPH_CHECK(!inputs.empty());
    return scatter_update::evaluate_scatter_update(outputs, inputs);
}

bool op::v3::ScatterUpdate::has_evaluate() const {
    NGRAPH_OP_SCOPE(v3_ScatterUpdate_has_evaluate);

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
