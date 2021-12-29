// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/scatter_nd_update.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/scatter_nd_update.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::v4::ScatterNDUpdate);

shared_ptr<ov::Node> ov::op::v4::ScatterNDUpdate::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v4_ScatterNDUpdate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v4::ScatterNDUpdate>(new_args.at(op::util::ScatterNDBase::INPUTS),
                                                new_args.at(op::util::ScatterNDBase::INDICES),
                                                new_args.at(op::util::ScatterNDBase::UPDATES));
}

namespace scatter {
namespace {
template <ov::element::Type_t ET>
bool evaluate(const ov::HostTensorPtr& arg0,
              const ov::HostTensorPtr& arg1,
              const ov::HostTensorPtr& arg2,
              const ov::HostTensorPtr& out) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::Shape params_shape = arg0->get_shape();
    ov::Shape indices_shape = arg1->get_shape();
    ov::Shape updates_shape = arg1->get_shape();
    const ov::Shape& out_shape(params_shape);
    out->set_shape(out_shape);

    if (arg1->get_element_type() == ov::element::i64) {
        ngraph::runtime::reference::scatterNdUpdate<T, int64_t>(arg0->get_data_ptr<ET>(),
                                                                arg1->get_data_ptr<int64_t>(),
                                                                arg2->get_data_ptr<ET>(),
                                                                out->get_data_ptr<ET>(),
                                                                arg0->get_shape(),
                                                                arg1->get_shape(),
                                                                arg2->get_shape());
    } else if (arg1->get_element_type() == ov::element::i32) {
        ngraph::runtime::reference::scatterNdUpdate<T, int32_t>(arg0->get_data_ptr<ET>(),
                                                                arg1->get_data_ptr<int32_t>(),
                                                                arg2->get_data_ptr<ET>(),
                                                                out->get_data_ptr<ET>(),
                                                                arg0->get_shape(),
                                                                arg1->get_shape(),
                                                                arg2->get_shape());
    } else {
        throw ov::Exception("Unexpected type");
    }

    return true;
}

bool evaluate_scatter(const ov::HostTensorPtr& arg0,
                      const ov::HostTensorPtr& arg1,
                      const ov::HostTensorPtr& arg2,
                      const ov::HostTensorPtr& out) {
    bool rc = true;

    switch (out->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_scatter, i32, arg0, arg1, arg2, out);
        NGRAPH_TYPE_CASE(evaluate_scatter, i64, arg0, arg1, arg2, out);
        NGRAPH_TYPE_CASE(evaluate_scatter, u32, arg0, arg1, arg2, out);
        NGRAPH_TYPE_CASE(evaluate_scatter, u64, arg0, arg1, arg2, out);
        NGRAPH_TYPE_CASE(evaluate_scatter, f16, arg0, arg1, arg2, out);
        NGRAPH_TYPE_CASE(evaluate_scatter, f32, arg0, arg1, arg2, out);
        NGRAPH_TYPE_CASE(evaluate_scatter, boolean, arg0, arg1, arg2, out);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace scatter

bool ov::op::v4::ScatterNDUpdate::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    NGRAPH_OP_SCOPE(v4_ScatterNDUpdate_evaluate);
    NGRAPH_CHECK(!inputs.empty());
    NGRAPH_CHECK(ngraph::validate_host_tensor_vector(inputs, 3));
    NGRAPH_CHECK(ngraph::validate_host_tensor_vector(outputs, 1));

    return scatter::evaluate_scatter(inputs[0], inputs[1], inputs[2], outputs[0]);
}

bool ov::op::v4::ScatterNDUpdate::has_evaluate() const {
    NGRAPH_OP_SCOPE(v4_ScatterNDUpdate_has_evaluate);

    switch (get_output_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::boolean:
        break;
    default:
        return false;
    }
    switch (get_input_element_type(1)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
        break;
    default:
        return false;
    }
    return true;
}
