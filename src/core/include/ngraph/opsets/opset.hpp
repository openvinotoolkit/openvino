// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <locale>
#include <map>
#include <mutex>
#include <set>

#include "ngraph/deprecated.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "openvino/opsets/opset.hpp"

namespace ngraph {
/// \brief Run-time opset information
class NGRAPH_API OpSet : public ov::OpSet {
    static std::mutex& get_mutex();

public:
    explicit OpSet(const ov::OpSet& opset);
    OpSet(const ngraph::OpSet& opset) = default;
    OpSet() = default;
    /// \brief Insert an op into the opset with a particular name and factory
    void insert(const std::string& name, const NodeTypeInfo& type_info, FactoryRegistry<Node>::Factory factory) {
        return ov::OpSet::insert(name, type_info, std::move(factory));
    }
    /// \brief Insert OP_TYPE into the opset with a special name and the default factory
    template <typename OP_TYPE>
    void insert(const std::string& name) {
        ov::OpSet::insert<OP_TYPE>(name);
    }

    /// \brief Insert OP_TYPE into the opset with the default name and factory
    template <typename OP_TYPE, typename std::enable_if<ngraph::HasTypeInfoMember<OP_TYPE>::value, bool>::type = true>
    void insert() {
        NGRAPH_SUPPRESS_DEPRECATED_START
        ov::OpSet::insert<OP_TYPE>(OP_TYPE::type_info.name);
        NGRAPH_SUPPRESS_DEPRECATED_END
    }

    template <typename OP_TYPE, typename std::enable_if<!ngraph::HasTypeInfoMember<OP_TYPE>::value, bool>::type = true>
    void insert() {
        ov::OpSet::insert<OP_TYPE>(OP_TYPE::get_type_info_static().name);
    }

    ngraph::FactoryRegistry<ngraph::Node>& get_factory_registry() {
        return m_factory_registry;
    }
};

const NGRAPH_API OpSet& get_opset1();
const NGRAPH_API OpSet& get_opset2();
const NGRAPH_API OpSet& get_opset3();
const NGRAPH_API OpSet& get_opset4();
const NGRAPH_API OpSet& get_opset5();
const NGRAPH_API OpSet& get_opset6();
const NGRAPH_API OpSet& get_opset7();
const NGRAPH_API OpSet& get_opset8();
const NGRAPH_API OpSet& get_opset9();
}  // namespace ngraph
