// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

namespace embedding_bag_offsets_sum_v3 {
template <ngraph::element::Type_t t1, ngraph::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ngraph::op::v3::EmbeddingBagOffsetsSum>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<t1>::value_type;
    using T2 = typename ngraph::element_type_traits<t2>::value_type;
    ngraph::runtime::reference::embeddingBagOffsetsSum<T1, T2>(
        inputs[0]->get_data_ptr<T1>(),
        inputs[1]->get_data_ptr<T2>(),
        inputs[2]->get_data_ptr<T2>(),
        inputs.size() > 3 ? inputs[3]->get_data_ptr<T2>() : nullptr,
        inputs.size() > 4 ? inputs[4]->get_data_ptr<T1>() : nullptr,
        outputs[0]->get_data_ptr<T1>(),
        ngraph::shape_size(inputs[1]->get_shape()),
        outputs[0]->get_shape());
}
}  // namespace embedding_bag_offsets_sum_v3

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v3::EmbeddingBagOffsetsSum>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[1]->get_element_type()) {
    case ngraph::element::Type_t::i32:
        embedding_bag_offsets_sum_v3::evaluate<ET, ngraph::element::Type_t::i32>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i64:
        embedding_bag_offsets_sum_v3::evaluate<ET, ngraph::element::Type_t::i64>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}