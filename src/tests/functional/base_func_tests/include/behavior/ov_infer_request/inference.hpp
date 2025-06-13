// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <string>
#include "common_test_utils/ov_plugin_cache.hpp"
#include "base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

struct OVInferReqInferParam {
    ov::Shape m_shape;
    ov::Tensor m_input_tensor;
    std::vector<float> m_expected;
    std::string m_test_name;
};

using OVInferRequestInferenceTestsParams = std::tuple<OVInferReqInferParam, std::string>;

namespace tensor_roi {
inline OVInferReqInferParam roi_nchw() {
    OVInferReqInferParam res;
    res.m_test_name = "roi_nchw";
    res.m_shape = Shape{1, 2, 3, 3};
    auto in_tensor = ov::Tensor(element::f32, Shape{1, 2, 5, 5});
    auto in_data = std::vector<float>{
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9,
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9,
            9, 8, 7, 6, 5,

            5, 6, 7, 8, 9,
            9, 8, 7, 6, 5,
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9,
            0, 1, 2, 3, 4
    };
    memcpy(in_tensor.data(), in_data.data(), in_data.size() * sizeof(float));
    res.m_input_tensor = ov::Tensor(in_tensor, Coordinate{0, 0, 1, 1}, Coordinate{1, 2, 4, 4});
    // Extracted 3x3 boxes, add 1 to each element
    res.m_expected = std::vector<float>{
            7, 8, 9,
            2, 3, 4,
            7, 8, 9,

            9, 8, 7,
            2, 3, 4,
            7, 8, 9,
    };
    return res;
}

inline OVInferReqInferParam roi_1d() {
    OVInferReqInferParam res;
    res.m_test_name = "roi_1d";
    res.m_shape = Shape{3};
    auto in_tensor = ov::Tensor(element::f32, Shape{5});
    auto in_data = std::vector<float>{10, 20, 30, 40, 50};
    memcpy(in_tensor.data(), in_data.data(), in_data.size() * sizeof(float));
    res.m_input_tensor = ov::Tensor(in_tensor, Coordinate{1}, Coordinate{4});
    res.m_expected = std::vector<float>{21, 31, 41};
    return res;
}

} // namespace tensor_roi

class OVInferRequestInferenceTests : public testing::WithParamInterface<OVInferRequestInferenceTestsParams>,
                                     public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<OVInferRequestInferenceTestsParams>& device_name);

protected:
    void SetUp() override;

    static std::shared_ptr<Model> create_n_inputs(size_t num, element::Type type,
                                                  const PartialShape& shape);

    std::shared_ptr<ov::Core> ie = utils::PluginCache::get().core();
    OVInferReqInferParam m_param;
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
