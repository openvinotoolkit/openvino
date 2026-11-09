// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/split.hpp"
#include "pass/isolate_precision_sensitive_outputs.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace {

using namespace ov::op;

std::shared_ptr<ov::Model> split_model(bool boundary,
                                       bool protect_source = true,
                                       bool protect_split = false,
                                       ov::element::Type type = ov::element::f32) {
    const auto x = std::make_shared<v0::Parameter>(type, ov::PartialShape{2, -1, 8});
    const auto source = std::make_shared<v0::Abs>(x);
    if (protect_source)
        ov::disable_conversion(source, ov::element::f16);
    ov::Output<ov::Node> value = source;
    if (boundary)
        value = std::make_shared<v0::Convert>(value, type);
    const auto split = std::make_shared<v1::Split>(value, v0::Constant::create(ov::element::i64, {}, {-1}), 2);
    split->set_friendly_name("split");
    if (protect_split)
        ov::disable_conversion(split, ov::element::f16);
    return std::make_shared<ov::Model>(ov::OutputVector{split->output(0), split->output(1), source},
                                       ov::ParameterVector{x});
}

class GGUFPrecisionBoundary : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::frontend::gguf::pass::IsolatePrecisionSensitiveOutputs>();
        manager.register_pass<ov::frontend::gguf::pass::IsolatePrecisionSensitiveOutputs>();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ACCURACY);
    }
};

TEST_F(GGUFPrecisionBoundary, IsolatesProducerWithoutChangingItsOtherConsumer) {
    model = split_model(false);
    model_ref = split_model(true);
}

TEST_F(GGUFPrecisionBoundary, KeepsUnprotectedProducer) {
    model = split_model(false, false);
}

TEST_F(GGUFPrecisionBoundary, KeepsProtectedSplit) {
    model = split_model(false, true, true);
}

TEST_F(GGUFPrecisionBoundary, KeepsExistingBoundary) {
    model = split_model(true);
}

TEST_F(GGUFPrecisionBoundary, KeepsOriginalFp16Graph) {
    model = split_model(false, true, false, ov::element::f16);
}

}  // namespace
