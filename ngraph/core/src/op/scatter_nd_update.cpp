// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/scatter_nd_update.hpp"
#include "itt.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::ScatterNDUpdate::type_info;

shared_ptr<Node> op::v3::ScatterNDUpdate::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v3_ScatterNDUpdate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v3::ScatterNDUpdate>(new_args.at(op::util::ScatterNDBase::INPUTS),
                                                new_args.at(op::util::ScatterNDBase::INDICES),
                                                new_args.at(op::util::ScatterNDBase::UPDATES));
}
