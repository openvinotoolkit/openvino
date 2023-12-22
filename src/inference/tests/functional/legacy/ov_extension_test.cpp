// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_extension_test.hpp"

#include "common_test_utils/file_utils.hpp"
#include "ie_iextension.h"
#include "openvino/op/op.hpp"
#include "openvino/runtime/core.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START

class CustomOldIdentity : public ov::op::Op {
public:
    OPENVINO_OP("Identity");

    CustomOldIdentity() = default;
    CustomOldIdentity(const ov::Output<ov::Node>& arg) : Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        if (new_args.size() != 1) {
            OPENVINO_THROW("Incorrect number of new arguments");
        }

        return std::make_shared<CustomOldIdentity>(new_args.at(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }
};

class TestTileOldExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

    void Unload() noexcept override {}

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        static std::map<std::string, ngraph::OpSet> opsets;
        if (opsets.empty()) {
            ngraph::OpSet opset;
            opset.insert<CustomOldIdentity>();
            opsets["extension"] = opset;
        }
        return opsets;
    }
};

TEST_F(OVExtensionTests, ReshapeIRWithOldExtension) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    core.add_extension(std::make_shared<TestTileOldExtension>());
    OPENVINO_SUPPRESS_DEPRECATED_END
    test();
}

OPENVINO_SUPPRESS_DEPRECATED_END