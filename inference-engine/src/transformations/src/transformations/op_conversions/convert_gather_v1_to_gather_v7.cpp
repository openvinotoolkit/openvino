// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_v1_to_gather_v7.hpp"

#include <ngraph/rt_info.hpp>

#include <numeric>

#include <ngraph/op/gather.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertGather1ToGather7, "ConvertGather1ToGather7", 0);

ngraph::pass::ConvertGather1ToGather7::ConvertGather1ToGather7() {
    MATCHER_SCOPE(ConvertGather1ToGather7);


    auto gather_v1 = pattern::wrap_type<ngraph::op::v1::Gather>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(gather_v1, matcher_name);
    register_matcher(m, callback);
}
