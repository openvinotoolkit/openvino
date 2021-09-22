// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>

#include "openvino/core/layout.hpp"
#include "openvino/core/preprocess/preprocess_steps.hpp"

namespace ov {
namespace preprocess {

/// \brief Preprocessing context passed to each preprocessing operation.
/// This is internal structure which is not shared to custom operations yet.
class PreprocessingContext {
public:
    explicit PreprocessingContext(const Layout& layout) : m_layout(layout) {}

    const Layout& layout() const {
        return m_layout;
    }

    Layout& layout() {
        return m_layout;
    }

private:
    Layout m_layout;
};

using InternalPreprocessOp =
    std::function<std::shared_ptr<ov::Node>(const std::vector<std::shared_ptr<ov::Node>>& nodes,
                                            PreprocessingContext& context)>;

/// \brief PreProcessStepsImpl - internal data structure
class PreProcessSteps::PreProcessStepsImpl {
public:
    void add_scale_impl(const std::vector<float>& values);
    void add_mean_impl(const std::vector<float>& values);
    void add_convert_impl(const element::Type& type);

    const std::list<std::tuple<InternalPreprocessOp, bool>>& actions() const {
        return m_actions;
    }
    std::list<std::tuple<InternalPreprocessOp, bool>>& actions() {
        return m_actions;
    }

private:
    std::list<std::tuple<InternalPreprocessOp, bool>> m_actions;
};

}  // namespace preprocess
}  // namespace ov
