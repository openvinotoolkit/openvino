// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "ngraph/op/util/max_pool_base.hpp"
#include "ngraph/shape.hpp"

#include <ngraph/validation_util.hpp>

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::MaxPoolBase, "MaxPoolBase", 8);

op::util::MaxPoolBase::MaxPoolBase(const Output<Node>& arg,
                                   const Strides& strides,
                                   const Shape& pads_begin,
                                   const Shape& pads_end,
                                   const Shape& kernel,
                                   op::RoundingType rounding_type,
                                   const op::PadType& auto_pad)
    : Op({arg})
    , m_kernel(kernel)
    , m_strides(strides)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_auto_pad(auto_pad)
    , m_rounding_type(rounding_type)
{
    constructor_validate_and_infer_types();
}
