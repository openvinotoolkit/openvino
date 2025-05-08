// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <numeric>
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using MOEExpertTestParams = std::tuple<ElementType>;  // input precision

class MOEExpertTest : public testing::WithParamInterface<MOEExpertTestParams>,
                      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MOEExpertTestParams>& obj) {
        ElementType inType;
        std::tie(inType) = obj.param;
        std::ostringstream result;

        result << "Prc=" << inType;
        return result.str();
    }
    void SetUp() override;

protected:
    // numpy randomint like inferface
    template<typename T>
    static std::vector<T> random(int start, int end, const std::vector<size_t>& shape, T scale = T{1}) {
        auto num = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
        std::vector<T> result(num);
        ov::test::utils::fill_data_random(result.data(), num, end - start, start);
        if (scale != T{1}) {
            std::transform(result.begin(), result.end(), result.begin(), [&scale](T x) {
                return x * scale;
            });
        }
        return result;
    }

    std::shared_ptr<ov::Model> BuildMoeExpert(ElementType inType,
                                              bool expected_pattern,
                                              int expert_num = 1,
                                              int topk = 8);

    void generate(float idx, size_t seq_length);
    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model);
    void check_op(const std::string& type_name, int expected_count);
    void prepare();

    size_t _expert_num = 4;
    size_t _topk = 2;
};
}  // namespace test
}  // namespace ov