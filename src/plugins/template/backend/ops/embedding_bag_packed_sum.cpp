// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "openvino/op/embeddingbag_packedsum.hpp"

namespace embedding_bag_packed_sum_v3 {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::v3::EmbeddingBagPackedSum>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ngraph::runtime::reference::embeddingBagPackedSum<T1, T2>(
        inputs[0]->get_data_ptr<T1>(),
        inputs[1]->get_data_ptr<T2>(),
        inputs.size() > 2 ? inputs[2]->get_data_ptr<T1>() : nullptr,
        outputs[0]->get_data_ptr<T1>(),
        inputs[1]->get_shape(),
        outputs[0]->get_shape());
}
}  // namespace embedding_bag_packed_sum_v3

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v3::EmbeddingBagPackedSum>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    switch (inputs[1]->get_element_type()) {
    case ov::element::Type_t::i32:
        embedding_bag_packed_sum_v3::evaluate<ET, ov::element::Type_t::i32>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i64:
        embedding_bag_packed_sum_v3::evaluate<ET, ov::element::Type_t::i64>(op, outputs, inputs);
        break;
    default:
        return false;
    }

    return true;
}
