// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/ctc_loss.hpp"

namespace ctc_loss_v4 {
template <ngraph::element::Type_t t1,
          ngraph::element::Type_t t2,
          typename std::enable_if<
              !std::is_floating_point<typename ngraph::element_type_traits<t1>::value_type>::value &&
                  !std::is_same<typename ngraph::element_type_traits<t1>::value_type, ngraph::bfloat16>::value &&
                  !std::is_same<typename ngraph::element_type_traits<t1>::value_type, ngraph::float16>::value,
              bool>::type = true>
inline void evaluate(const std::shared_ptr<ngraph::op::v4::CTCLoss>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    OPENVINO_ASSERT(false,
                    "The data type for logits is expected to be a floating point type. Got:",
                    ngraph::element::Type(t1));
}

template <ngraph::element::Type_t t1,
          ngraph::element::Type_t t2,
          typename std::enable_if<
              std::is_floating_point<typename ngraph::element_type_traits<t1>::value_type>::value ||
                  std::is_same<typename ngraph::element_type_traits<t1>::value_type, ngraph::bfloat16>::value ||
                  std::is_same<typename ngraph::element_type_traits<t1>::value_type, ngraph::float16>::value,
              bool>::type = true>
inline void evaluate(const std::shared_ptr<ngraph::op::v4::CTCLoss>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<t1>::value_type;
    using T2 = typename ngraph::element_type_traits<t2>::value_type;
    ngraph::runtime::reference::CTCLoss<T1, T2>(inputs[0]->get_data_ptr<T1>(),
                                                inputs[0]->get_shape(),
                                                inputs[1]->get_data_ptr<T2>(),
                                                inputs[2]->get_data_ptr<T2>(),
                                                inputs[3]->get_data_ptr<T2>(),
                                                inputs[4]->get_data_ptr<T2>(),
                                                op->get_preprocess_collapse_repeated(),
                                                op->get_ctc_merge_repeated(),
                                                op->get_unique(),
                                                outputs[0]->get_data_ptr<T1>());
}
}  // namespace ctc_loss_v4

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v4::CTCLoss>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[1]->get_element_type()) {
    case ngraph::element::Type_t::i32:
        ctc_loss_v4::evaluate<ET, ngraph::element::Type_t::i32>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i64:
        ctc_loss_v4::evaluate<ET, ngraph::element::Type_t::i64>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}