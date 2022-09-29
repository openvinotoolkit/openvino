// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>
#include "snippets/op/memory_access.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Load
 * @brief Generated during Lowering stage (convert_to_snippets_dialect) where explicit instructions should be emitted for data loading
 *        where number of elements to load is determined by "count" (Default value is "1" - to load one element)
 *        and memory offset for loading is determined by "offset" (Default value is "0" - to load starting from the first element)
 * @ingroup snippets
 */
class Load : public MemoryAccess {
public:
    OPENVINO_OP("Load", "SnippetsOpset");

    Load(const Output<Node>& x, const size_t count = 1lu, const size_t offset = 0lu);
    Load() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

/**
 * @interface LoadReshape
 * @brief It's just Load operation (and it's mapped on LoadEmitter during code generation) that allows to tweak
 *        shape propagation. We need it to keep correct shape propagation  when Transpose is decomposed to
 *        Load and Store. This is a temporary solution until tokenization of Reshape operation is supported.
 * @ingroup snippets
 */
class LoadReshape : public Load {
public:
    OPENVINO_OP("LoadReshape", "SnippetsOpset");
    LoadReshape(const Output<Node>& x, size_t count = 1lu, const size_t offset = 0lu, std::vector<size_t> order = {});
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
private:
    std::vector<size_t> m_order;
};
} // namespace op
} // namespace snippets
} // namespace ngraph
