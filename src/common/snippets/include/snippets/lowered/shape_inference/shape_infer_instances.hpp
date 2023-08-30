// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shape_inference.hpp"

namespace ov {
namespace snippets {
class entryNumpyBroadcasting : public IShapeInferSnippets {
public:
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};


template<class BroadcastOP>
class  BroadcastShapeInfer : public IShapeInferSnippets {
    VectorDims::value_type m_broadcasted_dim;
public:
    explicit BroadcastShapeInfer(const std::shared_ptr<Node>& n);
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};

class entryFirstPassThrough : public IShapeInferSnippets {
public:
    inline Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        OPENVINO_ASSERT(!input_shapes.empty(), "Empty Input shapes are not allowed for entryFirstPassthrough");
        return {{input_shapes[0].get()}, ShapeInferStatus::success};
    }
};

class entryEmpty : public IShapeInferSnippets {
public:
    inline Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        return {{}, ShapeInferStatus::success};
    }
};

class entrySingleElement : public IShapeInferSnippets {
public:
    inline Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        return {{{1}}, ShapeInferStatus::success};
    }
};

class SelectShapeInfer : public IShapeInferSnippets {
    ov::op::AutoBroadcastSpec m_broadcast_spec;
public:
    explicit SelectShapeInfer(const std::shared_ptr<Node>& n);
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};

} // namespace snippets
} // namespace ov
