// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {
namespace test {

class DenormalNullifyCheck : public SubgraphBaseTest {
protected:
std::unique_ptr<ov::AlignedBuffer> pConstStorage;

void validate() override {
    const auto& actualOutputs = get_plugin_outputs();
    ASSERT_FALSE(actualOutputs.empty());
    auto& outTensor = actualOutputs.front();
    ASSERT_EQ(ov::element::f32, outTensor.get_element_type()) << "Unexpected element type";
    const float* data = reinterpret_cast<const float*>(outTensor.data());
    bool hasDenormals = false;
    for (size_t i = 0; i < outTensor.get_size(); ++i) {
        if (std::abs(data[i]) >= std::numeric_limits<float>::denorm_min() &&
            std::abs(data[i]) < std::numeric_limits<float>::min()) {
            hasDenormals = true;
        }
    }
    ASSERT_FALSE(hasDenormals);
}


void SetUp() override {
    constexpr size_t alignment = 64; // bytes cache line size, to avoid denormals zeroing due to memory reallocation in the input node implementation
    const ov::Shape inpShape = {1, 24, 3, 3};
    targetStaticShapes.push_back({inpShape});
    targetDevice = ov::test::utils::DEVICE_CPU;

    const auto rtPrc = ov::element::f32;
    const auto elemsCount = shape_size(inpShape) * rtPrc.size();
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(rtPrc, ov::Shape(inpShape))};
    pConstStorage.reset(new ov::AlignedBuffer(elemsCount, alignment));

    auto constTensor = ov::Tensor(rtPrc, inpShape, pConstStorage->get_ptr());
    auto constNode = std::make_shared<ov::op::v0::Constant>(constTensor);
    ov::NodeVector input = {params[0], constNode};
    auto concat = std::make_shared<ov::op::v0::Concat>(input, 1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat->output(0))};

    function = std::make_shared<ov::Model>(results, params, "denormal_check");
}
};

TEST_F(DenormalNullifyCheck, smoke_CPU_Denormal_Check) {
    using indexInterval = std::pair<size_t, size_t>;
    size_t elemsCount = pConstStorage->size() / sizeof(float);
    const indexInterval intervals[] = {
        {0, elemsCount/2},
        {elemsCount/2, elemsCount},
        {0, elemsCount}
    };

    constexpr unsigned seed = 1u;
    constexpr unsigned denormalsCount = 15u;
    constexpr uint32_t denormalsRange = (0xffffffffu >> 9u) - 1;
    testing::internal::Random random(seed);
    auto randomRange = ov::test::utils::generateVector<ov::element::f32>(elemsCount, 10, -10);

    for (auto& interval : intervals) {
        auto randomIndices = ov::test::utils::generateVector<ov::element::u32>(denormalsCount, interval.second, interval.first);
        std::unordered_set<decltype(randomIndices)::value_type> randomIndexSet(randomIndices.begin(), randomIndices.end());
        for (size_t i = 0; i < elemsCount; ++i) {
            if (randomIndexSet.count(i)) {
                auto denormal = random.Generate(denormalsRange) + 1;
                float tmp;
                memcpy(&tmp, &denormal, sizeof(float));
                pConstStorage->get_ptr<float>()[i] = tmp;
            } else {
                pConstStorage->get_ptr<float>()[i] = randomRange[i];
            }
        }

        run();
    }
}

}  // namespace test
}  // namespace ov
