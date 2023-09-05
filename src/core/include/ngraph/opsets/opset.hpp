// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <locale>
#include <map>
#include <mutex>
#include <set>

#include "ngraph/deprecated.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "openvino/opsets/opset.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START
namespace ngraph {
/// \brief Run-time opset information
class NGRAPH_API OpSet : public ov::OpSet {
public:
    explicit OpSet(const ov::OpSet& opset);
    NGRAPH_SUPPRESS_DEPRECATED_START
    OpSet(const ngraph::OpSet& opset);
    NGRAPH_SUPPRESS_DEPRECATED_END
    OpSet() = default;
    /// \brief Insert an op into the opset with a particular name and factory
    void insert(const std::string& name, const NodeTypeInfo& type_info, FactoryRegistry<Node>::Factory factory) {
        ov::OpSet::insert(name, type_info, std::move(factory));
    }
    /// \brief Insert OP_TYPE into the opset with a special name and the default factory
    template <typename OP_TYPE>
    void insert(const std::string& name) {
        ov::OpSet::insert<OP_TYPE>(name);
    }

    /// \brief Insert OP_TYPE into the opset with the default name and factory
    template <typename OP_TYPE>
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
const NGRAPH_API OpSet& get_opset10();
const NGRAPH_API OpSet& get_opset11();
const NGRAPH_API OpSet& get_opset12();
const NGRAPH_API OpSet& get_opset13();
const NGRAPH_API std::map<std::string, std::function<const ngraph::OpSet&()>>& get_available_opsets();
}  // namespace ngraph
NGRAPH_SUPPRESS_DEPRECATED_END
