// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "./snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {
enum SoftmaxVersion { V1, V8 };
std::ostream &operator<<(std::ostream& os, const SoftmaxVersion& version);

class SoftmaxFunction : public SnippetsFunctionBase {
public:
    explicit SoftmaxFunction(const std::vector<PartialShape>& inputShapes,
                             int axis,
                             SoftmaxVersion softmax_version = SoftmaxVersion::V8)
        : SnippetsFunctionBase(inputShapes),
          axis(axis),
          softmax_version(softmax_version) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initLowered() const override;
    int axis;
    SoftmaxVersion softmax_version;
};

class AddSoftmaxFunction : public SnippetsFunctionBase {
public:
    explicit AddSoftmaxFunction(const std::vector<PartialShape>& inputShapes, int axis) : SnippetsFunctionBase(inputShapes), axis(axis) {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    int axis;
};

class TransposeSoftmaxFunction : public SnippetsFunctionBase {
public:
    explicit TransposeSoftmaxFunction(const std::vector<PartialShape>& inputShapes, const std::vector<int64_t>& order, const int64_t axis)
            : SnippetsFunctionBase(inputShapes), m_order(order), m_axis(axis) {
        OPENVINO_ASSERT(input_shapes.size() > 0, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    std::vector<int64_t> m_order;
    int64_t m_axis;
};

class TransposeSoftmaxEltwiseFunction : public TransposeSoftmaxFunction {
public:
    explicit TransposeSoftmaxEltwiseFunction(const std::vector<PartialShape>& inputShapes, const std::vector<int64_t>& order, const int64_t axis)
            : TransposeSoftmaxFunction(inputShapes, order, axis) {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
