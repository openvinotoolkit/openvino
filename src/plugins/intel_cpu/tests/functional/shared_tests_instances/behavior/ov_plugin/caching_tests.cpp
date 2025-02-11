// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"

using namespace ov;
using namespace ov::test::behavior;

namespace {
    static const std::vector<ov::element::Type> precisionsCPU = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i32,
            ov::element::i64,
            ov::element::i8,
            ov::element::u8,
            ov::element::i16,
            ov::element::u16,
    };

    static const std::vector<ov::element::Type> floatPrecisionsCPU = {
            ov::element::f32,
            ov::element::f16,
    };

    static const std::vector<std::size_t> batchSizesCPU = {
            1, 2
    };

    static const std::vector<ov::element::Type> precisionsCPUInternal = {
            ov::element::f32
    };

    static const std::vector<std::size_t> batchSizesCPUInternal = {
            1
    };

    static std::shared_ptr<Model> simple_function_non_max_suppression_internal(element::Type, size_t) {
        auto boxes = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = ov::op::v0::Constant::create(element::i32, Shape{1}, {10});
        auto iou_threshold = ov::op::v0::Constant::create(element::f32, Shape{1}, {0.75});
        auto score_threshold = ov::op::v0::Constant::create(element::f32, Shape{1}, {0.7});
        auto nms = std::make_shared<ov::op::internal::NonMaxSuppressionIEInternal>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, 0, true, element::i32);
        auto res = std::make_shared<ov::op::v0::Result>(nms);
        auto func = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
        return func;
    }

    static std::shared_ptr<Model> simple_function_matrix_nms_internal(element::Type, size_t) {
        auto boxes = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        ov::op::v8::MatrixNms::Attributes attr;
        // convert_precision does not support internal op 'NmsStaticShapeIE'
        attr.output_type = element::i32;
        auto nms = std::make_shared<ov::op::internal::NmsStaticShapeIE<ov::op::v8::MatrixNms>>(boxes, scores, attr);
        auto res = std::make_shared<ov::op::v0::Result>(nms);
        auto func = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
        return func;
    }

    static std::shared_ptr<Model> simple_function_multiclass_nms_internal(element::Type, size_t) {
        auto boxes = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        ov::op::util::MulticlassNmsBase::Attributes attr;
        attr.output_type = element::i32;
        auto nms = std::make_shared<ov::op::internal::MulticlassNmsIEInternal>(boxes, scores, attr);
        auto res = std::make_shared<ov::op::v0::Result>(nms);
        auto func = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
        return func;
    }

    static std::vector<ovModelWithName> internal_functions_cpu() {
        std::vector<ovModelWithName> funcs = {
            ovModelWithName { simple_function_non_max_suppression_internal, "NonMaxSuppressionIEInternal"},
            ovModelWithName { simple_function_matrix_nms_internal, "NmsStaticShapeIE_MatrixNms"},
            ovModelWithName { simple_function_multiclass_nms_internal, "MulticlassNmsIEInternal"},
        };
        return funcs;
    }

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_CPU, CompileModelCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(CompileModelCacheTestBase::getNumericAnyTypeFunctions()),
                                    ::testing::ValuesIn(precisionsCPU),
                                    ::testing::ValuesIn(batchSizesCPU),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                    ::testing::Values(ov::AnyMap{})),
                            CompileModelCacheTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_CPU_Float, CompileModelCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(CompileModelCacheTestBase::getFloatingPointOnlyFunctions()),
                                    ::testing::ValuesIn(floatPrecisionsCPU),
                                    ::testing::ValuesIn(batchSizesCPU),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                    ::testing::Values(ov::AnyMap{})),
                            CompileModelCacheTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_CPU_Internal, CompileModelCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(internal_functions_cpu()),
                                    ::testing::ValuesIn(precisionsCPUInternal),
                                    ::testing::ValuesIn(batchSizesCPUInternal),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                    ::testing::Values(ov::AnyMap{})),
                            CompileModelCacheTestBase::getTestCaseName);

    const std::vector<ov::AnyMap> CpuConfigs = {
        {ov::num_streams(2)},
    };
    const std::vector<std::string> TestCpuTargets = {
        ov::test::utils::DEVICE_CPU,
    };
    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_CPU,
                             CompileModelLoadFromMemoryTestBase,
                             ::testing::Combine(::testing::ValuesIn(TestCpuTargets), ::testing::ValuesIn(CpuConfigs)),
                             CompileModelLoadFromMemoryTestBase::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_CPU,
                             CompileModelLoadFromFileTestBase,
                             ::testing::Combine(::testing::ValuesIn(TestCpuTargets), ::testing::ValuesIn(CpuConfigs)),
                             CompileModelLoadFromMemoryTestBase::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_CPU,
                             CompileModelCacheRuntimePropertiesTestBase,
                             ::testing::Combine(::testing::ValuesIn(TestCpuTargets), ::testing::ValuesIn(CpuConfigs)),
                             CompileModelCacheRuntimePropertiesTestBase::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_CPU,
                             CompileModelLoadFromCacheTest,
                             ::testing::Combine(::testing::ValuesIn(TestCpuTargets), ::testing::ValuesIn(CpuConfigs)),
                             CompileModelLoadFromCacheTest::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_CPU,
                             CompileModelWithCacheEncryptionTest,
                             ::testing::ValuesIn(TestCpuTargets),
                             CompileModelWithCacheEncryptionTest::getTestCaseName);
} // namespace
