// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <legacy/ngraph_ops/crop_ie.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "backend/gna_limitations.hpp"
#include "common/gna_target.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "ops/copy.hpp"
#include "ov_models/builders.hpp"
#include "transformations/insert_copy_layer.hpp"

using namespace ov::intel_gna::limitations;
using namespace ov::intel_gna::target;

namespace testing {

typedef std::tuple<DeviceVersion,  // Device version
                   size_t,         // Concat axis
                   size_t          // input number
                   >
    InsertCopyTestParams;

class InsertCopyLayerTest : public ov::test::TestsCommon, public ::testing::WithParamInterface<InsertCopyTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InsertCopyTestParams>& obj) {
        DeviceVersion device_ver;
        size_t axis, inputs_num;
        std::tie(device_ver, axis, inputs_num) = obj.param;

        std::ostringstream result;
        result << DeviceToString(device_ver) << "_";
        result << "inputsNum=" << inputs_num << "_";
        result << "axis=" << axis;

        return result.str();
    }
    void SetUp() override;
    virtual void Validate();
    virtual void Run();

public:
    std::shared_ptr<ngraph::Function> m_func, m_ref_func;
    DeviceVersion m_device_ver;
    size_t m_axis, m_inputs_num;
};

void InsertCopyLayerTest::Validate() {
    ASSERT_NO_THROW(check_rt_info(m_func));

    auto result = compare_functions(m_func, m_ref_func);
    ASSERT_TRUE(result.first);
}

void InsertCopyLayerTest::SetUp() {
    std::tie(m_device_ver, m_axis, m_inputs_num) = this->GetParam();
    Limitations::init(m_device_ver);
}

void InsertCopyLayerTest::Run() {
    Validate();
}

//      [Parameter]            [Parameter]
//        \   /                   \   /
//        [Add]                  [Add]
//      /   | ..\              /   | ..\
//      \   | ../             | [Copy] [Copy]
//       [Concat]              \   | ../
//          |                   [Concat]
//       [Result]                  |
//                              [Result]
class InsertCopyLayerConcatTest : public InsertCopyLayerTest {
public:
    void SetUp() override {
        ngraph::Shape input_shape{10};
        InsertCopyLayerTest::SetUp();

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
            auto add = std::make_shared<ngraph::opset8::Add>(params, params);
            ngraph::OutputVector concat_inputs;
            for (int i = 0; i < m_inputs_num; ++i) {
                concat_inputs.push_back(add);
            }
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, m_axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
            auto add = std::make_shared<ngraph::opset8::Add>(params, params);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(add);
            ngraph::OutputVector concat_inputs = {};
            for (int i = 0; i < m_inputs_num - 1; ++i) {
                concat_inputs.push_back(copy);
            }
            concat_inputs.push_back(add);
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, m_axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            m_ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }
    }

    void Validate() override {
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(m_func);

        InsertCopyLayerTest::Validate();
    }
};

//      [Parameter]            [Parameter]
//           |                     |
//       [Split]                [Split]
//      /   | ..\              /   | ..\
//      \   | ../             | [Copy] [Copy]
//       [Concat]              \   | ../
//          |                   [Concat]
//       [Result]                  |
//                               [Result]
class InsertCopyLayerSplitConcatTest : public InsertCopyLayerTest {
public:
    void SetUp() override {
        ngraph::Shape input_shape{256};
        InsertCopyLayerTest::SetUp();

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
            auto split = ngraph::builder::makeSplit(params, ngraph::element::i64, m_inputs_num, m_axis);

            ngraph::OutputVector concat_inputs;
            for (int i = 0; i < m_inputs_num; ++i) {
                concat_inputs.push_back(split->output(i));
            }
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, m_axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
            auto split = ngraph::builder::makeSplit(params, ngraph::element::i64, m_inputs_num, m_axis);

            ngraph::OutputVector concat_inputs;
            int copy_layer_interval =
                (Limitations::get_instance()->get_memory_alignment() / Limitations::kBytesPerSplitElement) *
                m_inputs_num / input_shape[0];

            for (int i = 0; i < m_inputs_num; ++i) {
                if (m_inputs_num == 1 || (i % copy_layer_interval == 0))
                    concat_inputs.push_back(std::make_shared<ov::intel_gna::op::Copy>(split->output(i)));
                else
                    concat_inputs.push_back(split->output(i));
            }
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, m_axis);

            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            m_ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }
    }
    void Validate() override {
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::InsertCopyBeforeConcatLayer>();
        m.run_passes(m_func);

        InsertCopyLayerTest::Validate();
    }
};

class TransformationTestsBase : public ov::test::TestsCommon,
                                public ::testing::WithParamInterface<std::tuple<DeviceVersion>> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<DeviceVersion>>& obj) {
        DeviceVersion device_ver;
        std::tie(device_ver) = obj.param;

        std::ostringstream result;
        result << DeviceToString(device_ver);

        return result.str();
    }

    void SetUp() override {
        std::tie(m_device_ver) = this->GetParam();
        Limitations::init(m_device_ver);
    }

    void TearDown() override {
        m_func.reset();
    }

    void RunPasses(ngraph::pass::Manager& m) {
        m.run_passes(m_func);
    }

    void Validate(const std::shared_ptr<ngraph::Function>& f_ref) {
        ASSERT_NO_THROW(check_rt_info(m_func));
        auto result1 = compare_functions(m_func, f_ref);
        ASSERT_TRUE(result1.first);
    }

    void Validate(const std::shared_ptr<ngraph::Function>& f_ref1, const std::shared_ptr<ngraph::Function>& f_ref2) {
        ASSERT_NO_THROW(check_rt_info(m_func));

        auto result1 = compare_functions(m_func, f_ref1);
        auto result2 = compare_functions(m_func, f_ref2);
        ASSERT_TRUE(result1.first || result2.first);
    }

public:
    DeviceVersion m_device_ver;
    std::shared_ptr<ngraph::Function> m_func;
};

//      [Parameter]            [Parameter]
//        \     /       =>         |
//       [Concat]                [Copy]
//           |                    \  /
//        [Result]              [Concat]
//                                  |
//                               [Result]
using InsertCopyLayerMultiParamConcatTest = TransformationTestsBase;
TEST_P(InsertCopyLayerMultiParamConcatTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    size_t axis = 0;
    ngraph::Shape in_shape{10};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        ngraph::OutputVector concat_inputs{params, params};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        m_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(params);

        ngraph::OutputVector concat_inputs{copy, copy};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleMultiConnectedLayerToConcatAndMemory>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerMultiParamConcatTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

//      [Parameter]              [Parameter]
//       /       \                /       \
//    [Reshape][Reshape]      [Reshape][Reshape]
//         \     /       =>      |         |
//         [Concat]            [Copy]   [Copy]
//            |                     \     /
//         [Result]                [Concat]
//                                     |
//                                 [Result]
using InsertCopyLayerMultiParamNFLConcatTest = TransformationTestsBase;
TEST_P(InsertCopyLayerMultiParamNFLConcatTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    size_t axis = 0;
    ngraph::Shape shape = {1, 1, 2, 4};
    ngraph::Shape in_shape = {1, 2, 4};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape1 = ov::op::util::reshapeTo(params, shape);
        auto reshape2 = ov::op::util::reshapeTo(params, shape);
        ngraph::OutputVector concat_inputs{reshape1, reshape2};

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        m_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape1 = ov::op::util::reshapeTo(params, shape);
        auto reshape2 = ov::op::util::reshapeTo(params, shape);
        auto copy1 = std::make_shared<ov::intel_gna::op::Copy>(reshape1);
        auto copy2 = std::make_shared<ov::intel_gna::op::Copy>(reshape2);

        ngraph::OutputVector concat_inputs{copy1, copy2};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleMultiConnectedLayerToConcatAndMemory>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerMultiParamNFLConcatTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

//      [Parameter]                [Parameter]
//       /       \                  /       \
//    [Reshape][Reshape]        [Reshape][Reshape]
//    /     |   \/   |           /     \      |
// [Result] |   /\   |    => [Result] [Copy [Copy]
//      [Concat] [Concat]              |   \/   |
//        |        |                   |   /\   |
//      [Result] [Result]           [Concat] [Concat]
//                                     |        |
//                                  [Result] [Result]
using InsertCopyLayerMultiParamMultiNFLConcatTest = TransformationTestsBase;
TEST_P(InsertCopyLayerMultiParamMultiNFLConcatTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    size_t axis = 0;
    ngraph::Shape shape = {1, 1, 2, 4};
    ngraph::Shape in_shape = {1, 2, 4};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape1 = ov::op::util::reshapeTo(params, shape);
        auto reshape2 = ov::op::util::reshapeTo(params, shape);
        ngraph::OutputVector concat_inputs{reshape1, reshape2};

        auto concat1 = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto concat2 = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result1 = std::make_shared<ngraph::opset8::Result>(concat1);
        auto result2 = std::make_shared<ngraph::opset8::Result>(concat2);
        auto result3 = std::make_shared<ngraph::opset8::Result>(reshape1);
        m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                                    ngraph::ParameterVector{params},
                                                    "Concat");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape1 = ov::op::util::reshapeTo(params, shape);
        auto reshape2 = ov::op::util::reshapeTo(params, shape);
        auto copy1 = std::make_shared<ov::intel_gna::op::Copy>(reshape1);
        auto copy2 = std::make_shared<ov::intel_gna::op::Copy>(reshape2);

        ngraph::OutputVector concat_inputs{copy1, copy2};
        auto concat1 = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto concat2 = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result1 = std::make_shared<ngraph::opset8::Result>(concat1);
        auto result2 = std::make_shared<ngraph::opset8::Result>(concat2);
        auto result3 = std::make_shared<ngraph::opset8::Result>(reshape1);
        ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                                      ngraph::ParameterVector{params},
                                                      "Concat");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleMultiConnectedLayerToConcatAndMemory>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerMultiParamMultiNFLConcatTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

//  [Parameter][Constant]  [Parameter][Constant]
//      \      |      /       \       |       /
//         [Concat]            \   [Copy]    /
//             |         =>     \     |     /
//         [Result]               [Concat]
//                                    |
//                                 [Result]
using InsertCopyLayerMultiConstConcatTest = TransformationTestsBase;
TEST_P(InsertCopyLayerMultiConstConcatTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func1, ref_func2;
    size_t axis = 0;
    ngraph::Shape in_shape{10};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto constant = std::make_shared<ngraph::opset8::Constant>(ngraph::element::i64, in_shape);

        ngraph::OutputVector concat_inputs{params, constant, constant};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        m_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto constant = std::make_shared<ngraph::opset8::Constant>(ngraph::element::i64, in_shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(constant);

        ngraph::OutputVector concat_inputs{params, copy, constant};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func1 =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto constant = std::make_shared<ngraph::opset8::Constant>(ngraph::element::i64, in_shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(constant);

        ngraph::OutputVector concat_inputs{params, constant, copy};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func2 =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::InsertCopyBeforeConcatLayer>();

    RunPasses(m);
    Validate(ref_func1, ref_func2);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerMultiConstConcatTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter]     [Parameter]
//   \    /          \    /
//   [Add]            [Add]
//    \  /              |  \
//  [Concat]   =>    [Copy] |
//     |                \  /
//  [Result]           [Concat]
//                        |
//                     [Result]
using InsertCopyLayerMultiLayerConcatTest = TransformationTestsBase;
TEST_P(InsertCopyLayerMultiLayerConcatTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func1, ref_func2;
    size_t axis = 0;
    ngraph::Shape in_shape{10};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto add = std::make_shared<ngraph::opset8::Add>(params, params);
        ngraph::OutputVector concat_inputs{add, add};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        m_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto add = std::make_shared<ngraph::opset8::Add>(params, params);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(add);

        ngraph::OutputVector concat_inputs{copy, add};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func1 =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto add = std::make_shared<ngraph::opset8::Add>(params, params);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(add);

        ngraph::OutputVector concat_inputs{add, copy};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func2 =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleMultiConnectedLayerToConcatAndMemory>();

    RunPasses(m);
    // Transformation is based on outputs order and insert copy layer in one of the branches,
    // so this is right, that we have two different result graph based on output order.
    Validate(ref_func1, ref_func1);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerMultiLayerConcatTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter]     [Constant]     [Parameter]    [Constant]
//     |    \          |             |    \         |
//  [Assign] \    [ReadValue]     [Copy]  [Copy]  [ReadValue]
//            \     /                |       \     /
//             [Add]        => [Assign]        [Add]
//                |                              |
//            [Result]                        [Result]
using InsertCopyLayerMultiLayerNFLConcatTest = TransformationTestsBase;
TEST_P(InsertCopyLayerMultiLayerNFLConcatTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func1, ref_func2;
    size_t axis = 0;
    ngraph::Shape shape = {1, 1, 2, 4};
    ngraph::Shape in_shape = {1, 2, 4};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto add = std::make_shared<ngraph::opset8::Add>(params, params);
        auto reshape1 = ov::op::util::reshapeTo(add, shape);
        auto reshape2 = ov::op::util::reshapeTo(add, shape);
        ngraph::OutputVector concat_inputs{reshape1, reshape2};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        m_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto add = std::make_shared<ngraph::opset8::Add>(params, params);
        auto reshape1 = ov::op::util::reshapeTo(add, shape);
        auto reshape_copy = std::make_shared<ov::intel_gna::op::Copy>(reshape1);
        auto reshape2 = ov::op::util::reshapeTo(add, shape);

        ngraph::OutputVector concat_inputs{reshape_copy, reshape2};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func1 =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto add = std::make_shared<ngraph::opset8::Add>(params, params);
        auto reshape1 = ov::op::util::reshapeTo(add, shape);
        auto reshape2 = ov::op::util::reshapeTo(add, shape);
        auto reshape_copy = std::make_shared<ov::intel_gna::op::Copy>(reshape2);

        ngraph::OutputVector concat_inputs{reshape1, reshape_copy};
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func2 =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleMultiConnectedLayerToConcatAndMemory>();

    RunPasses(m);
    // Transformation is based on outputs order and insert copy layer in one of the branches,
    // so this is right, that we have two different result graph based on output order.
    Validate(ref_func1, ref_func2);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerMultiLayerNFLConcatTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter]     [Constant]     [Parameter]    [Constant]
//     |    \          |             |    \         |
//  [Assign] \    [ReadValue]     [Copy]  [Copy]  [ReadValue]
//            \     /                |       \     /
//             [Add]        => [Assign]        [Add]
//                |                              |
//            [Result]                        [Result]
using InsertCopyLayerMultiParamMemoryTest = TransformationTestsBase;
TEST_P(InsertCopyLayerMultiParamMemoryTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    ngraph::Shape in_shape{10};
    const std::string variable_name("variable_id");

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{in_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, in_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto add = std::make_shared<ngraph::opset8::Add>(input, read_value);
        auto result = std::make_shared<ngraph::opset8::Result>(add);
        auto assign = std::make_shared<ngraph::opset8::Assign>(input, variable);
        assign->add_control_dependency(read_value);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        m_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{in_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, in_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto copy1 = std::make_shared<ov::intel_gna::op::Copy>(input);
        auto add = std::make_shared<ngraph::opset8::Add>(copy1, read_value);
        auto result = std::make_shared<ngraph::opset8::Result>(add);
        auto copy2 = std::make_shared<ov::intel_gna::op::Copy>(input);
        auto assign = std::make_shared<ngraph::opset8::Assign>(copy2, variable);
        assign->add_control_dependency(read_value);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleMultiConnectedLayerToConcatAndMemory>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerMultiParamMemoryTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter]     [Constant]     [Parameter]    [Constant]
//     |    \          |             |    \         |
//  [Assign] \    [ReadValue]     [Copy]  [Copy]  [ReadValue]
//            \     /                |       \     /
//            [Concat]        => [Assign]     [Concat]
//                |                              |
//            [Result]                        [Result]
using InsertCopyLayerMultiParamConcatMemoryTest = TransformationTestsBase;
TEST_P(InsertCopyLayerMultiParamConcatMemoryTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    ngraph::Shape in_shape{10};
    size_t axis = 0;
    const std::string variable_name("variable_id");

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{in_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, in_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(input, variable);
        assign->add_control_dependency(read_value);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input, read_value}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        m_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{in_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto copy1 = std::make_shared<ov::intel_gna::op::Copy>(input);
        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, in_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(copy1, variable);
        assign->add_control_dependency(read_value);
        auto copy2 = std::make_shared<ov::intel_gna::op::Copy>(input);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{copy2, read_value}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleMultiConnectedLayerToConcatAndMemory>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerMultiParamConcatMemoryTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

//   [Parameter]     [Constant]     [Parameter]    [Constant]
//     /      \         |             /      \         |
// [Reshape][Reshape][ReadValue] [Reshape][Reshape][ReadValue]
//     |       \       /             |        |        /
// [Assign]    [Concat]        =>   [Copy]   [Copy]   /
//                 |                  |        \     /
//              [Result]          [Assign]     [Concat]
//                                                |
//                                             [Result]
using InsertCopyLayerMultiParamNFLConcatMemoryTest = TransformationTestsBase;
TEST_P(InsertCopyLayerMultiParamNFLConcatMemoryTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    ngraph::Shape in_shape = {1, 2, 4};
    ngraph::Shape shape1 = {1, 1, 2, 4};
    ngraph::Shape shape2 = {2, 4};
    size_t axis = 0;
    const std::string variable_name("variable_id");

    {
        auto variable =
            std::make_shared<ngraph::Variable>(ov::op::util::VariableInfo{shape2, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape1 = ov::op::util::reshapeTo(input, shape1);
        auto reshape2 = ov::op::util::reshapeTo(input, shape2);

        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, shape2, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(reshape1, variable);
        assign->add_control_dependency(read_value);

        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{reshape2, read_value}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        m_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    {
        auto variable =
            std::make_shared<ngraph::Variable>(ov::op::util::VariableInfo{shape2, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape1 = ov::op::util::reshapeTo(input, shape1);
        auto reshape2 = ov::op::util::reshapeTo(input, shape2);
        auto copy1 = std::make_shared<ov::intel_gna::op::Copy>(reshape1);
        auto copy2 = std::make_shared<ov::intel_gna::op::Copy>(reshape2);

        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, shape2, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(copy1, variable);
        assign->add_control_dependency(read_value);

        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{copy2, read_value}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleMultiConnectedLayerToConcatAndMemory>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerMultiParamNFLConcatMemoryTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter]    [Constant]         [Parameter]    [Constant]
//     |               |                 |               |
// [Reshape]      [ReadValue]        [Reshape]      [ReadValue]
//     |              /                  |              /
// [CropIE]          /       =>       [CropIE]         /
//     |    \       /                    |    \       /
// [Assign]   [Mul]                    [Copy]    [Mul]
//              |                        |        |
//           [Result]                [Assign]  [Result]
using InsertCopyLayerMultiLayerConcatMemoryTest = TransformationTestsBase;
TEST_P(InsertCopyLayerMultiLayerConcatMemoryTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    std::vector<int64_t> axes = {0, 1, 2, 3};
    std::vector<int64_t> dim = {1, 1, 2, 2};
    std::vector<int64_t> offset = {0, 0, 0, 0};
    ngraph::Shape shape = {1, 1, 2, 4};
    ngraph::Shape in_shape = {1, 2, 4};
    ngraph::Shape out_shape = {1, 1, 2, 2};
    const std::string variable_name("variable_id");

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{out_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape = ov::op::util::reshapeTo(input, shape);
        auto crop = std::make_shared<ngraph::op::CropIE>(reshape, axes, dim, offset);

        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, out_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto mul = std::make_shared<ngraph::opset8::Multiply>(crop, read_value);
        auto assign = std::make_shared<ngraph::opset8::Assign>(crop, variable);
        assign->add_control_dependency(read_value);
        auto result = std::make_shared<ngraph::opset8::Result>(mul);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        m_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{out_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape = ov::op::util::reshapeTo(input, shape);
        auto crop = std::make_shared<ngraph::op::CropIE>(reshape, axes, dim, offset);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(crop);

        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, out_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto mul = std::make_shared<ngraph::opset8::Multiply>(crop, read_value);
        auto assign = std::make_shared<ngraph::opset8::Assign>(copy, variable);
        assign->add_control_dependency(read_value);
        auto result = std::make_shared<ngraph::opset8::Result>(mul);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::InsertCopyBeforeAssignLayer>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerMultiLayerConcatMemoryTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter]    [Constant]         [Parameter]    [Constant]
//     |               |                 |               |
// [Reshape]      [ReadValue]        [Reshape]      [ReadValue]
//     |               /                 |               /
// [CropIE]           /              [CropIE]           /
//     |             /       =>          |             /
// [Reshape]        /                [Reshape]        /
//     |    \      /                     |    \      /
// [Assign]   [Add]                   [Copy]    [Add]
//              |                        |        |
//           [Result]                [Assign] [Result]
using InsertCopyLayerCropMemoryTest = TransformationTestsBase;
TEST_P(InsertCopyLayerCropMemoryTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    std::vector<int64_t> axes = {0, 1, 2, 3};
    std::vector<int64_t> dim = {1, 1, 2, 2};
    std::vector<int64_t> offset = {0, 0, 0, 0};
    ngraph::Shape shape1 = {1, 1, 2, 4};
    ngraph::Shape shape2 = {1, 1, 2, 2};
    ngraph::Shape in_shape = {1, 2, 4};
    const std::string variable_name("variable_id");

    {
        auto variable =
            std::make_shared<ngraph::Variable>(ov::op::util::VariableInfo{shape2, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape1 = ov::op::util::reshapeTo(input, shape1);
        auto crop = std::make_shared<ngraph::op::CropIE>(reshape1, axes, dim, offset);
        auto reshape2 = ov::op::util::reshapeTo(crop, shape2);

        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, shape2, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto add = std::make_shared<ngraph::opset8::Add>(reshape2, read_value);
        auto assign = std::make_shared<ngraph::opset8::Assign>(reshape2, variable);
        assign->add_control_dependency(read_value);
        auto result = std::make_shared<ngraph::opset8::Result>(add);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        m_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    {
        auto variable =
            std::make_shared<ngraph::Variable>(ov::op::util::VariableInfo{shape2, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape1 = ov::op::util::reshapeTo(input, shape1);
        auto crop = std::make_shared<ngraph::op::CropIE>(reshape1, axes, dim, offset);
        auto reshape2 = ov::op::util::reshapeTo(crop, shape2);

        auto copy = std::make_shared<ov::intel_gna::op::Copy>(reshape2);
        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, shape2, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto add = std::make_shared<ngraph::opset8::Add>(reshape2, read_value);
        auto assign = std::make_shared<ngraph::opset8::Assign>(copy, variable);
        assign->add_control_dependency(read_value);
        auto result = std::make_shared<ngraph::opset8::Result>(add);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::InsertCopyBeforeAssignLayer>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerCropMemoryTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter] [Constant]      [Parameter]    [Constant]
//     |            |               |            |
// [Reshape]  [ReadValue]        [Reshape]   [ReadValue]
//     \        /                    \         /
//      [Concat]          =>          [Concat]
//         |                              |
//      [Split]                        [Split]
//       /    \                         /    \
// [Assign]  [Result]              [Сopy]  [Result]
//                                    |
//                                 [Assign]
using InsertCopyLayerCropNFLMemoryTest = TransformationTestsBase;
TEST_P(InsertCopyLayerCropNFLMemoryTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    ngraph::Shape in_shape{10};
    size_t axis = 0;
    const std::string variable_name("variable_id");

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{in_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, in_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input, read_value}, axis);
        auto axis_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto split = std::make_shared<ngraph::opset8::Split>(concat, axis_const, 2);
        auto result = std::make_shared<ngraph::opset8::Result>(split);
        auto assign = std::make_shared<ngraph::opset8::Assign>(split, variable);
        assign->add_control_dependency(read_value);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        m_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{in_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, in_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input, read_value}, axis);
        auto axis_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto split = std::make_shared<ngraph::opset8::Split>(concat, axis_const, 2);
        auto result = std::make_shared<ngraph::opset8::Result>(split);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(split);
        auto assign = std::make_shared<ngraph::opset8::Assign>(copy, variable);
        assign->add_control_dependency(read_value);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::InsertCopyBeforeAssignLayer>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerCropNFLMemoryTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter1][Parameter2][Constant]  [Parameter1][Parameter2][Constant]
//     |           /           |            |            /         |
// [Reshape]      /        [ReadValue]   [Reshape]      /     [ReadValue]
//     \         /            /              \         /          /
//       [Concat]            /         =>     [Concat]           /
//       /      \           /                  /    \           /
// [Assign]      \         /              [Сopy]     \         /
//                  [Add]                    |          [Add]
//                    |                   [Assign]        |
//                 [Result]                           [Result]
using InsertCopyLayerConcatMemoryTest = TransformationTestsBase;
TEST_P(InsertCopyLayerConcatMemoryTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    ngraph::Shape in_shape = {1, 2, 4};
    ngraph::Shape out_shape = {2, 2, 4};
    size_t axis = 0;
    const std::string variable_name("variable_id");

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{out_shape, ngraph::element::i64, variable_name});
        auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input2}, axis);

        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, out_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(concat, variable);
        assign->add_control_dependency(read_value);
        auto add = std::make_shared<ngraph::opset8::Add>(concat, read_value);
        auto result = std::make_shared<ngraph::opset8::Result>(add);

        ngraph::ParameterVector params = {input1, input2};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        m_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{out_shape, ngraph::element::i64, variable_name});
        auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input2}, axis);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(concat);

        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, out_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(copy, variable);
        assign->add_control_dependency(read_value);
        auto add = std::make_shared<ngraph::opset8::Add>(concat, read_value);
        auto result = std::make_shared<ngraph::opset8::Result>(add);

        ngraph::ParameterVector params = {input1, input2};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::InsertCopyBeforeAssignLayer>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerConcatMemoryTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter1][Parameter2][Constant]  [Parameter1][Parameter2][Constant]
//     |           /           |            |            /         |
// [Reshape]      /        [ReadValue]   [Reshape]      /     [ReadValue]
//     \         /            /              \         /          /
//       [Concat]            /         =>     [Concat]           /
//           |              /                      |            /
//       [Reshape]         /                   [Reshape]       /
//       /      \         /                      /    \       /
// [Assign]      \       /                  [Сopy]     \     /
//                 [Add]                       |        [Add]
//                   |                      [Assign]      |
//                [Result]                             [Result]
using InsertCopyLayerConcatNFLMemoryTest = TransformationTestsBase;
TEST_P(InsertCopyLayerConcatNFLMemoryTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    ngraph::Shape shape = {1, 2, 2, 4};
    ngraph::Shape in_shape = {1, 2, 4};
    size_t axis = 0;
    const std::string variable_name("variable_id");

    {
        auto variable =
            std::make_shared<ngraph::Variable>(ov::op::util::VariableInfo{shape, ngraph::element::i64, variable_name});
        auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input2}, axis);
        auto reshape = ov::op::util::reshapeTo(concat, shape);

        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(reshape, variable);
        assign->add_control_dependency(read_value);
        auto add = std::make_shared<ngraph::opset8::Add>(reshape, read_value);
        auto result = std::make_shared<ngraph::opset8::Result>(add);

        ngraph::ParameterVector params = {input1, input2};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        m_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    {
        auto variable =
            std::make_shared<ngraph::Variable>(ov::op::util::VariableInfo{shape, ngraph::element::i64, variable_name});
        auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input2}, axis);
        auto reshape = ov::op::util::reshapeTo(concat, shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(reshape);

        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(copy, variable);
        assign->add_control_dependency(read_value);
        auto add = std::make_shared<ngraph::opset8::Add>(reshape, read_value);
        auto result = std::make_shared<ngraph::opset8::Result>(add);

        ngraph::ParameterVector params = {input1, input2};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::InsertCopyBeforeAssignLayer>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerConcatNFLMemoryTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter] [Constant]      [Parameter] [Constant]
//     |           |               |           |
//  [Split]   [ReadValue]        [Split]   [ReadValue]
//   /     \      /              /    \       /
// [Assign][Concat]           [Сopy]   [Concat]
//            |                 |         |
//          [Result]         [Assign   [Result]
using InsertCopyLayerSplitMemoryTest = TransformationTestsBase;
TEST_P(InsertCopyLayerSplitMemoryTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    ngraph::Shape in_shape{10};
    ngraph::Shape out_shape{5};
    size_t axis = 0;
    const std::string variable_name("variable_id");

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{out_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto split = ngraph::builder::makeSplit(input, ngraph::element::i64, 1, axis);
        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, out_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(split, variable);
        assign->add_control_dependency(read_value);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{split, read_value}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        m_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{out_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto split = ngraph::builder::makeSplit(input, ngraph::element::i64, 1, axis);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(split);
        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, out_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(copy, variable);
        assign->add_control_dependency(read_value);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{split, read_value}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::InsertCopyBeforeAssignLayer>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerSplitMemoryTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter] [Constant]    [Parameter] [Constant]
//    |            |              |           |
// [Split]    [ReadValue]      [Split]   [ReadValue]
//    |    \      /              |   \        /
// [Reshape]\    /            [Reshape]\     /
//    |      \  /                |      \   /
// [Assign] [Concat]          [Сopy]  [Concat]
//             |                 |       |
//          [Result]         [Assign   [Result]
using InsertCopyLayerSplitNFLMemoryTest = TransformationTestsBase;
TEST_P(InsertCopyLayerSplitNFLMemoryTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    ngraph::Shape in_shape{10};
    ngraph::Shape shape{1, 5};
    ngraph::Shape out_shape{5};
    size_t axis = 0;
    const std::string variable_name("variable_id");

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{out_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto split = ngraph::builder::makeSplit(input, ngraph::element::i64, 2, axis);
        auto reshape = ov::op::util::reshapeTo(split, shape);
        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, out_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(reshape, variable);
        assign->add_control_dependency(read_value);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{split, read_value}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        m_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    {
        auto variable = std::make_shared<ngraph::Variable>(
            ov::op::util::VariableInfo{out_shape, ngraph::element::i64, variable_name});
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto split = ngraph::builder::makeSplit(input, ngraph::element::i64, 2, axis);
        auto reshape = ov::op::util::reshapeTo(split, shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(reshape);
        auto init_value = ngraph::builder::makeConstant(ngraph::element::i64, out_shape, std::vector<size_t>{0});
        auto read_value = std::make_shared<ngraph::opset8::ReadValue>(init_value, variable);
        auto assign = std::make_shared<ngraph::opset8::Assign>(copy, variable);
        assign->add_control_dependency(read_value);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{split, read_value}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);

        ngraph::ParameterVector params = {input};
        ngraph::ResultVector results = {result};
        ngraph::SinkVector sinks = {assign};
        ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::InsertCopyBeforeAssignLayer>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerSplitNFLMemoryTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter]                [Parameter]
//      |                          |
//  [Reshape]                  [Reshape]
//      |                          |
//  [CropIE] [Constant]   =>   [CropIE]  [Constant]
//      \       /                  |          /
//       [Concat]               [Copy]       /
//           |                     \        /
//       [Result]                   [Concat]
//                                     |
//                                  [Result]
using InsertCopyLayerCropConcatTest = TransformationTestsBase;
TEST_P(InsertCopyLayerCropConcatTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    size_t axis = 0;
    std::vector<int64_t> axes = {0, 1, 2, 3};
    std::vector<int64_t> dim = {1, 1, 2, 2};
    std::vector<int64_t> offset = {0, 0, 0, 0};
    ngraph::Shape shape = {1, 1, 2, 4};
    ngraph::Shape in_shape = {1, 2, 4};
    ngraph::Shape out_shape = {1, 1, 2, 2};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape = ov::op::util::reshapeTo(params, shape);
        auto crop = std::make_shared<ngraph::op::CropIE>(reshape, axes, dim, offset);
        auto const_value = ngraph::builder::makeConstant(ngraph::element::i64, out_shape, std::vector<size_t>{1});
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{crop, const_value}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        m_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape = ov::op::util::reshapeTo(params, shape);
        auto crop = std::make_shared<ngraph::op::CropIE>(reshape, axes, dim, offset);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(crop);
        auto const_value = ngraph::builder::makeConstant(ngraph::element::i64, out_shape, std::vector<size_t>{1});
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{copy, const_value}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::InsertCopyBeforeConcatLayer>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerCropConcatTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter]      [Parameter]
//      |               |
//  [Reshape]  =>     [Copy]
//      |               |
//   [Result]       [Reshape]
//                      |
//                   [Result]
using InsertCopyLayerNonfuncTest = TransformationTestsBase;
TEST_P(InsertCopyLayerNonfuncTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    std::vector<int64_t> axes = {0, 1, 2, 3};
    std::vector<int64_t> dim = {1, 1, 2, 2};
    std::vector<int64_t> offset = {0, 0, 0, 0};
    ngraph::Shape shape = {1, 1, 2, 4};
    ngraph::Shape in_shape = {1, 2, 4};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape = ov::op::util::reshapeTo(params, shape);
        auto result = std::make_shared<ngraph::opset8::Result>(reshape);
        m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                    ngraph::ParameterVector{params},
                                                    "nonfunc");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(params);
        auto reshape = ov::op::util::reshapeTo(copy, shape);
        auto result = std::make_shared<ngraph::opset8::Result>(reshape);
        ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                      ngraph::ParameterVector{params},
                                                      "nonfunc");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleNonFunctionalSubgraphs>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerNonfuncTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

//    [Parameter]        [Parameter]
//      /     \               |
// [Reshape][Reshape] =>    [Copy]
//     |        |            /  \
//  [Result] [Result]  [Reshape][Reshape]
//                        |         |
//                     [Result] [Result]
using InsertCopyLayerNonfuncTwoSubgraphsTest = TransformationTestsBase;
TEST_P(InsertCopyLayerNonfuncTwoSubgraphsTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    std::vector<int64_t> axes = {0, 1, 2, 3};
    std::vector<int64_t> dim = {1, 1, 2, 2};
    std::vector<int64_t> offset = {0, 0, 0, 0};
    ngraph::Shape shape = {1, 1, 2, 4};
    ngraph::Shape in_shape = {1, 2, 4};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape1 = ov::op::util::reshapeTo(params, shape);
        auto reshape2 = ov::op::util::reshapeTo(params, shape);
        auto result1 = std::make_shared<ngraph::opset8::Result>(reshape1);
        auto result2 = std::make_shared<ngraph::opset8::Result>(reshape2);
        m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                    ngraph::ParameterVector{params},
                                                    "nonfunc");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(params);
        auto reshape1 = ov::op::util::reshapeTo(copy, shape);
        auto reshape2 = ov::op::util::reshapeTo(copy, shape);
        auto result1 = std::make_shared<ngraph::opset8::Result>(reshape1);
        auto result2 = std::make_shared<ngraph::opset8::Result>(reshape2);
        ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                      ngraph::ParameterVector{params},
                                                      "nonfunc");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleNonFunctionalSubgraphs>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerNonfuncTwoSubgraphsTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

//   [Parameter]        [Parameter]
//        |                  |
//    [Reshape]           [Copy]
//     /     \               |
//  [Result] [Result]    [Reshape]
//                        /      \
//                     [Result] [Result]
using InsertCopyLayerNonfuncTwoResultsTest = TransformationTestsBase;
TEST_P(InsertCopyLayerNonfuncTwoResultsTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    std::vector<int64_t> axes = {0, 1, 2, 3};
    std::vector<int64_t> dim = {1, 1, 2, 2};
    std::vector<int64_t> offset = {0, 0, 0, 0};
    ngraph::Shape shape = {1, 1, 2, 4};
    ngraph::Shape in_shape = {1, 2, 4};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape = ov::op::util::reshapeTo(params, shape);
        auto result1 = std::make_shared<ngraph::opset8::Result>(reshape);
        auto result2 = std::make_shared<ngraph::opset8::Result>(reshape);
        m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                    ngraph::ParameterVector{params},
                                                    "nonfunc");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(params);
        auto reshape = ov::op::util::reshapeTo(copy, shape);
        auto result1 = std::make_shared<ngraph::opset8::Result>(reshape);
        auto result2 = std::make_shared<ngraph::opset8::Result>(reshape);
        ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                      ngraph::ParameterVector{params},
                                                      "nonfunc");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleNonFunctionalSubgraphs>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerNonfuncTwoResultsTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter]        [Parameter]
//     |                   |
// [Reshape]            [Reshape]
//     |    \    =>        |     \
//  [Relu]  [Reshape]    [Relu]  [Copy]
//     |        |          |        |
//  [Result] [Result]   [Result] [Reshape]
//                                  |
//                               [Result]
using InsertCopyLayerNFLBranchTest = TransformationTestsBase;
TEST_P(InsertCopyLayerNFLBranchTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    std::vector<int64_t> axes = {0, 1, 2, 3};
    std::vector<int64_t> dim = {1, 1, 2, 2};
    std::vector<int64_t> offset = {0, 0, 0, 0};
    ngraph::Shape shape = {1, 1, 2, 4};
    ngraph::Shape in_shape = {1, 2, 4};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape = ov::op::util::reshapeTo(params, shape);
        auto reshape2 = ov::op::util::reshapeTo(reshape, shape);
        auto result = std::make_shared<ngraph::opset8::Result>(reshape2);

        auto relu = std::make_shared<ngraph::opset8::Relu>(reshape);
        auto result_relu = std::make_shared<ngraph::opset8::Result>(relu);

        m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result, result_relu},
                                                    ngraph::ParameterVector{params},
                                                    "nonfunc");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape = ov::op::util::reshapeTo(params, shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(reshape);
        auto reshape2 = ov::op::util::reshapeTo(copy, shape);
        auto result = std::make_shared<ngraph::opset8::Result>(reshape2);

        auto relu = std::make_shared<ngraph::opset8::Relu>(reshape);
        auto result_relu = std::make_shared<ngraph::opset8::Result>(relu);

        ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result, result_relu},
                                                      ngraph::ParameterVector{params},
                                                      "nonfunc");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleNonFunctionalSubgraphs>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerNFLBranchTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter]        [Parameter]
//     |                   |
// [Reshape]            [Reshape]
//     |    \      =>      |     \
//  [Relu]  [Reshape]    [Relu]  [Copy]
//     |        |          |        |
// [Reshape] [Result]   [Reshape] [Reshape]
//     |                   |          |
//  [Result]            [Result]   [Result]
using InsertCopyLayerNFLvsFLSubgraphTest = TransformationTestsBase;
TEST_P(InsertCopyLayerNFLvsFLSubgraphTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    std::vector<int64_t> axes = {0, 1, 2, 3};
    std::vector<int64_t> dim = {1, 1, 2, 2};
    std::vector<int64_t> offset = {0, 0, 0, 0};
    ngraph::Shape shape = {1, 1, 2, 4};
    ngraph::Shape in_shape = {1, 2, 4};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape = ov::op::util::reshapeTo(params, shape);
        auto result = std::make_shared<ngraph::opset8::Result>(reshape);

        auto relu = std::make_shared<ngraph::opset8::Relu>(params);
        auto reshape2 = ov::op::util::reshapeTo(relu, shape);
        auto result_relu = std::make_shared<ngraph::opset8::Result>(reshape2);

        m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result, result_relu},
                                                    ngraph::ParameterVector{params},
                                                    "nonfunc");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(params);
        auto reshape = ov::op::util::reshapeTo(copy, shape);
        auto result = std::make_shared<ngraph::opset8::Result>(reshape);

        auto relu = std::make_shared<ngraph::opset8::Relu>(params);
        auto reshape2 = ov::op::util::reshapeTo(relu, shape);
        auto result_relu = std::make_shared<ngraph::opset8::Result>(reshape2);

        ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result, result_relu},
                                                      ngraph::ParameterVector{params},
                                                      "nonfunc");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::HandleNonFunctionalSubgraphs>();

    RunPasses(m);
    Validate(ref_func);
}
INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerNFLvsFLSubgraphTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

// [Parameter]              [Parameter]
//     |                         |
//  [Split]                  [Split]
//     |               =>       |
// [Reshape] [Constant]      [Reshape]
//     \      /                 |
//    [Concat]                [Copy] [Constant]
//       |                       \      /
//    [Result]                   [Concat]
//                                  |
//                               [Result]
using InsertCopyLayerSplitNFLConcatTest = TransformationTestsBase;
TEST_P(InsertCopyLayerSplitNFLConcatTest, CompareWithRefs) {
    std::shared_ptr<ngraph::Function> ref_func;
    ngraph::Shape input_shape{1, 2, 4};
    ngraph::Shape shape{1, 1, 2, 4};
    size_t axis = 0;

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
        auto split = ngraph::builder::makeSplit(params, ngraph::element::i64, 1, axis);
        auto reshape = ov::op::util::reshapeTo(split->output(0), shape);
        auto const_value = ngraph::builder::makeConstant(ngraph::element::i64, shape, std::vector<size_t>{1});
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{reshape, const_value}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        m_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }
    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
        auto split = ngraph::builder::makeSplit(params, ngraph::element::i64, 1, axis);
        auto reshape = ov::op::util::reshapeTo(split->output(0), shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(reshape);
        auto const_value = ngraph::builder::makeConstant(ngraph::element::i64, shape, std::vector<size_t>{1});
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{copy, const_value}, axis);

        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{params}, "Concat");
    }

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::InsertCopyBeforeConcatLayer>();

    RunPasses(m);
    Validate(ref_func);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerSplitNFLConcatTest,
                         ::testing::Values(DeviceVersion::GNA3_0, DeviceVersion::GNA3_5, DeviceVersion::GNA3_6),
                         TransformationTestsBase::getTestCaseName);

TEST_P(InsertCopyLayerConcatTest, CompareWithRefs) {
    Run();
}

TEST_P(InsertCopyLayerSplitConcatTest, CompareWithRefs) {
    Run();
}

const size_t axis = 0;
const std::vector<size_t> inputCounts = {1, 64, 128, 256};

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerConcatTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<DeviceVersion>{DeviceVersion::GNA3_0,
                                                                                           DeviceVersion::GNA3_5,
                                                                                           DeviceVersion::GNA3_6}),
                                            ::testing::Values(axis),
                                            ::testing::ValuesIn(inputCounts)),
                         InsertCopyLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertCopyLayerSplitConcatTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<DeviceVersion>{DeviceVersion::GNA3_0,
                                                                                           DeviceVersion::GNA3_5,
                                                                                           DeviceVersion::GNA3_6}),
                                            ::testing::Values(axis),
                                            ::testing::ValuesIn(inputCounts)),
                         InsertCopyLayerTest::getTestCaseName);

}  // namespace testing
