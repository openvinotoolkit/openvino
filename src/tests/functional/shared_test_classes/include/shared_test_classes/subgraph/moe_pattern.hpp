// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

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
    template <typename IT, typename T>
    static void strided_iota(IT first, size_t n, T value, T stride) {
        for (size_t i = 0; i < n; i++) {
            *first++ = value;
            value += stride;
        }
    }

    static std::vector<uint8_t> genList(size_t len, size_t start, size_t end) {
        std::vector<uint8_t> result(len);
        if (start == end) {
            for (size_t i = 0; i < len; i++) {
                result[i] = start;
            }
        } else {
            for (size_t i = 0; i < len; i++) {
                result[i] = (start + i) % end;
            }
        }
        return result;
    }

    static std::vector<float> genList(size_t len, float start, float stride) {
        std::vector<float> result(len);
        strided_iota(result.begin(), len, start, stride);
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