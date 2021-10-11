// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/gather_nd_base.hpp"

#include <ngraph/validation_util.hpp>

#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/shape.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::util::GatherNDBase);

ov::op::util::GatherNDBase::GatherNDBase(const Output<Node>& data, const Output<Node>& indices, const size_t batch_dims)
    : Op({data, indices}),
      m_batch_dims(batch_dims) {
    constructor_validate_and_infer_types();
}
