// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ngraph_test_utils.hpp>
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        PartialShape,  // Input shape 0
        PartialShape,  // Input shape 1
        PartialShape,  // Input shape 2
        size_t,        // Add input index
        bool           // Constant input
> MulAddToFMAParams;

class MulAddToFMATests : public TransformationTestsF, public testing::WithParamInterface<MulAddToFMAParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MulAddToFMAParams> obj);

protected:
    void SetUp() override;

    std::shared_ptr<SnippetsFunctionBase> snippets_function;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
