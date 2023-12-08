// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"

namespace ov {
namespace test {

template<typename T>
class AlignedBufferWrapper {
public:
    AlignedBufferWrapper(size_t size, size_t alignment) {
        _buffer.reset(new ngraph::runtime::AlignedBuffer(size * sizeof(T), alignment));
    }
    AlignedBufferWrapper(const AlignedBufferWrapper&) = delete;
    AlignedBufferWrapper& operator=(const AlignedBufferWrapper&) = delete;
    AlignedBufferWrapper(AlignedBufferWrapper&&) = default;
    AlignedBufferWrapper& operator=(AlignedBufferWrapper&&) = default;

    T* get_ptr() {
        return _buffer->get_ptr<T>();
    }

    size_t size() const {
        return _buffer->size() / sizeof(T);
    }
private:
    std::unique_ptr<ngraph::runtime::AlignedBuffer> _buffer = nullptr;
};

class DenormalNullifyCheck : public SubgraphBaseTest {
protected:
std::unique_ptr<AlignedBufferWrapper<float>> pConstStorage;

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

    const auto elemsCount = shape_size(inpShape);
    const auto rtPrc = ov::element::f32;
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(rtPrc, ov::Shape(inpShape))};
    pConstStorage.reset(new AlignedBufferWrapper<float>(elemsCount, alignment));

    auto constTensor = std::make_shared<ngraph::HostTensor>(rtPrc, inpShape, pConstStorage->get_ptr());
    auto constNode = std::make_shared<ov::op::v0::Constant>(constTensor);
    ov::NodeVector input = {params[0], constNode};
    auto concat = std::make_shared<ov::op::v0::Concat>(input, 1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat->output(0))};

    function = std::make_shared<ov::Model>(results, params, "denormal_check");
}
};

TEST_F(DenormalNullifyCheck, smoke_CPU_Denormal_Check) {
    using indexInterval = std::pair<size_t, size_t>;
    size_t elemsCount = pConstStorage->size();
    const indexInterval intervals[] = {
        {0, elemsCount/2},
        {elemsCount/2, elemsCount},
        {0, elemsCount}
    };

    constexpr unsigned seed = 1u;
    constexpr unsigned denormalsCount = 15u;
    constexpr uint32_t denormalsRange = (0xffffffffu >> 9u) - 1;
    testing::internal::Random random(seed);
    auto randomRange = NGraphFunctions::Utils::generateVector<ov::element::f32>(elemsCount, 10, -10);

    for (auto& interval : intervals) {
        auto randomIndices = NGraphFunctions::Utils::generateVector<ov::element::u32>(denormalsCount, interval.second, interval.first);
        std::unordered_set<decltype(randomIndices)::value_type> randomIndexSet(randomIndices.begin(), randomIndices.end());
        for (size_t i = 0; i < elemsCount; ++i) {
            if (randomIndexSet.count(i)) {
                auto denormal = random.Generate(denormalsRange) + 1;
                float tmp;
                memcpy(&tmp, &denormal, sizeof(float));
                pConstStorage->get_ptr()[i] = tmp;
            } else {
                pConstStorage->get_ptr()[i] = randomRange[i];
            }
        }

        run();
    }
}

}  // namespace test
}  // namespace ov
