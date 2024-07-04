// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_with_reference.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "transformations/init_node_info.hpp"

FrontEndConversionWithReferenceTestsF::FrontEndConversionWithReferenceTestsF()
    : comparator(FunctionsComparator::no_default()) {
    comparator.enable(FunctionsComparator::CmpValues::NODES);
    comparator.enable(FunctionsComparator::CmpValues::PRECISIONS);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
    comparator.enable(FunctionsComparator::CmpValues::SUBGRAPH_DESCRIPTORS);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

void FrontEndConversionWithReferenceTestsF::SetUp() {
    manager.register_pass<ov::pass::InitNodeInfo>();
}

void FrontEndConversionWithReferenceTestsF::TearDown() {
    OPENVINO_ASSERT(model != nullptr, "Test Model is not initialized.");
    OPENVINO_ASSERT(model_ref != nullptr, "Reference Test Model is not initialized.");

    manager.run_passes(model);
    OV_ASSERT_NO_THROW(check_rt_info(model));

    auto res = comparator.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}
