// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

class ExperimentalDetectronPriorGridGeneratorTestParam {
public:
    ov::op::v6::ExperimentalDetectronPriorGridGenerator::Attributes attributes;
    std::vector<InputShape> inputShapes;
};

typedef std::tuple<
    ExperimentalDetectronPriorGridGeneratorTestParam,
    std::pair<std::string, std::vector<ov::runtime::Tensor>>,
    ElementType,                // Network precision
    std::string                 // Device name>;
> ExperimentalDetectronPriorGridGeneratorTestParams;

class ExperimentalDetectronPriorGridGeneratorLayerTest :
        public testing::WithParamInterface<ExperimentalDetectronPriorGridGeneratorTestParams>,
        virtual public SubgraphBaseTest {
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronPriorGridGeneratorTestParams>& obj);

    template <class T>
    static ov::runtime::Tensor createTensor(
            const ov::element::Type& element_type,
            const Shape& shape,
            const std::vector<T>& values,
            const size_t size = 0) {
        const size_t real_size = size ? size : values.size() * sizeof(T) / element_type.size();
        ov::runtime::Tensor tensor { element_type, shape };
        std::memcpy(tensor.data(), values.data(), std::min(real_size * element_type.size(), sizeof(T) * values.size()));
        return tensor;
    }
};
} // namespace subgraph
} // namespace test
} // namespace ov
