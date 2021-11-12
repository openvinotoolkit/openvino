// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/ops.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/utils/compare_results.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
namespace utils {

namespace {
void compare(const std::shared_ptr<ov::Node> node,
             size_t port,
             const ov::runtime::Tensor &expected,
             const ov::runtime::Tensor &actual,
             double absThreshold,
             double relThreshold) {
    ov::test::utils::compare(expected, actual, absThreshold, relThreshold);
}

template<typename T>
void compareResults(const std::shared_ptr<ov::Node> node,
                    size_t port,
                    const ov::runtime::Tensor &expected,
                    const ov::runtime::Tensor &actual,
                    double absThreshold,
                    double relThreshold) {
    return compare(ngraph::as_type_ptr<T>(node), port, expected, actual, absThreshold, relThreshold);
}

} // namespace

CompareMap getCompareMap() {
    CompareMap compareMap{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), compareResults<NAMESPACE::NAME>},

#include "ngraph/opsets/opset1_tbl.hpp"
#include "ngraph/opsets/opset2_tbl.hpp"
#include "ngraph/opsets/opset3_tbl.hpp"
#include "ngraph/opsets/opset4_tbl.hpp"
#include "ngraph/opsets/opset5_tbl.hpp"
#include "ngraph/opsets/opset6_tbl.hpp"

#undef NGRAPH_OP
    };
    return compareMap;
}

} // namespace utils
} // namespace test
} // namespace ov
