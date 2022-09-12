// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"

#include "ngraph/log.hpp"
#include "ngraph/ops.hpp"
#include "openvino/core/node.hpp"
#include "opset_factory.hpp"

ngraph::OpSet::OpSet(const ov::OpSet& opset) : ov::OpSet(opset) {}

const ngraph::FactoryRegistry<ngraph::Node>& ngraph::OpSet::get_factory_registry() {
    NGRAPH_SUPPRESS_DEPRECATED_START
    for (const auto& builder : m_factory->get_builders()) {
        m_factory_registry.register_factory(builder.first, builder.second);
    }
    return m_factory_registry;
    NGRAPH_SUPPRESS_DEPRECATED_END
}

/**
 * @brief Run-time opset information
 *
 * Helper class which purpose is to provide full capabilities
 * of conditional compilation for operation set. To do so,
 * use
 * ```
 * opset.get_factory()->registerImplIfRequired(<args>)
 * opset.insert_type_info_only(<args>)
 * ```
 * instead of `opset.insert(<args>)`.
 * Inline use of `registerImplIfRequired()`
 * is a limitation of CC implementation.
 */
class CCOpSet : public ov::OpSet {
public:
    CCOpSet(const std::string& name) {
        m_factory = std::unique_ptr<ov::opset::Factory>(new ov::opset::Factory{name});
    }

    void insert_type_info_only(const std::string& name, const ov::NodeTypeInfo& type_info) {
        m_op_types.insert(type_info);
        m_name_type_info_map[name] = type_info;
        m_case_insensitive_type_info_map[to_upper_name(name)] = type_info;
    }

    const std::unique_ptr<ov::opset::Factory>& get_factory() {
        return m_factory;
    }
};

ov::OpSet::OpSet() {
    m_factory = std::unique_ptr<ov::opset::Factory>(new ov::opset::Factory{"default_factory"});
}

ov::OpSet::OpSet(const ov::OpSet& opset) {
    m_op_types = opset.m_op_types;
    m_name_type_info_map = opset.m_name_type_info_map;
    m_case_insensitive_type_info_map = opset.m_case_insensitive_type_info_map;
    m_factory = std::unique_ptr<ov::opset::Factory>(new ov::opset::Factory(*opset.m_factory));
}

ov::OpSet ov::OpSet::operator=(const ov::OpSet& opset) {
    m_op_types = opset.m_op_types;
    m_name_type_info_map = opset.m_name_type_info_map;
    m_case_insensitive_type_info_map = opset.m_case_insensitive_type_info_map;
    m_factory = std::unique_ptr<ov::opset::Factory>(new ov::opset::Factory(*opset.m_factory));
    return *this;
}

ov::OpSet::~OpSet() {}

std::mutex& ov::OpSet::get_mutex() {
    static std::mutex opset_mutex;
    return opset_mutex;
}

void ov::OpSet::insert(const std::string& name,
                       const ov::NodeTypeInfo& type_info,
                       const std::function<ov::Node*()>& builder) {
    std::lock_guard<std::mutex> guard(get_mutex());

    m_op_types.insert(type_info);
    m_name_type_info_map[name] = type_info;
    m_case_insensitive_type_info_map[to_upper_name(name)] = type_info;
    m_factory->register_type(type_info, std::move(builder));
}

ov::Node* ov::OpSet::create(const std::string& name) const {
    auto type_info_it = m_name_type_info_map.find(name);
    if (type_info_it == m_name_type_info_map.end()) {
        NGRAPH_WARN << "Couldn't create operator of type: " << name << " . Operation not registered in opset.";
        return nullptr;
    }
    return m_factory->create(type_info_it->second);
}

ov::Node* ov::OpSet::create_insensitive(const std::string& name) const {
    auto type_info_it = m_case_insensitive_type_info_map.find(to_upper_name(name));
    return type_info_it == m_case_insensitive_type_info_map.end() ? nullptr : m_factory->create(type_info_it->second);
}

const ov::OpSet& ov::get_opset1() {
    static CCOpSet opset("opset1");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
    using namespace opset;
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.get_factory()->registerImplIfRequired(opset_factory, NAME, NAMESPACE::NAME::get_type_info_static(), opset.get_default_builder<NAMESPACE::NAME>()); \
                                          opset.insert_type_info_only(NAMESPACE::NAME::get_type_info_static().name, NAMESPACE::NAME::get_type_info_static());
#include "openvino/opsets/opset1_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset2() {
    static CCOpSet opset("opset2");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
    using namespace opset;
    
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.get_factory()->registerImplIfRequired(opset_factory, NAME, NAMESPACE::NAME::get_type_info_static(), opset.get_default_builder<NAMESPACE::NAME>()); \
                                          opset.insert_type_info_only(NAMESPACE::NAME::get_type_info_static().name, NAMESPACE::NAME::get_type_info_static());
#include "openvino/opsets/opset2_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset3() {
    static CCOpSet opset("opset3");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
    using namespace opset;
    
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.get_factory()->registerImplIfRequired(opset_factory, NAME, NAMESPACE::NAME::get_type_info_static(), opset.get_default_builder<NAMESPACE::NAME>()); \
                                          opset.insert_type_info_only(NAMESPACE::NAME::get_type_info_static().name, NAMESPACE::NAME::get_type_info_static());
#include "openvino/opsets/opset3_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset4() {
    static CCOpSet opset("opset4");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
    using namespace opset;
    
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.get_factory()->registerImplIfRequired(opset_factory, NAME, NAMESPACE::NAME::get_type_info_static(), opset.get_default_builder<NAMESPACE::NAME>()); \
                                          opset.insert_type_info_only(NAMESPACE::NAME::get_type_info_static().name, NAMESPACE::NAME::get_type_info_static());
#include "openvino/opsets/opset4_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset5() {
    static CCOpSet opset("opset5");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
    using namespace opset;
    
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.get_factory()->registerImplIfRequired(opset_factory, NAME, NAMESPACE::NAME::get_type_info_static(), opset.get_default_builder<NAMESPACE::NAME>()); \
                                          opset.insert_type_info_only(NAMESPACE::NAME::get_type_info_static().name, NAMESPACE::NAME::get_type_info_static());
#include "openvino/opsets/opset5_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset6() {
    static CCOpSet opset("opset6");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
    using namespace opset;
    using namespace opset;
    
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.get_factory()->registerImplIfRequired(opset_factory, NAME, NAMESPACE::NAME::get_type_info_static(), opset.get_default_builder<NAMESPACE::NAME>()); \
                                          opset.insert_type_info_only(NAMESPACE::NAME::get_type_info_static().name, NAMESPACE::NAME::get_type_info_static());
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset7() {
    static CCOpSet opset("opset7");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
    using namespace opset;
    
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.get_factory()->registerImplIfRequired(opset_factory, NAME, NAMESPACE::NAME::get_type_info_static(), opset.get_default_builder<NAMESPACE::NAME>()); \
                                          opset.insert_type_info_only(NAMESPACE::NAME::get_type_info_static().name, NAMESPACE::NAME::get_type_info_static());
#include "openvino/opsets/opset7_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset8() {
    static CCOpSet opset("opset8");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
    using namespace opset;
    
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.get_factory()->registerImplIfRequired(opset_factory, NAME, NAMESPACE::NAME::get_type_info_static(), opset.get_default_builder<NAMESPACE::NAME>()); \
                                          opset.insert_type_info_only(NAMESPACE::NAME::get_type_info_static().name, NAMESPACE::NAME::get_type_info_static());
#include "openvino/opsets/opset8_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset9() {
    static CCOpSet opset("opset9");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
    using namespace opset;
    
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.get_factory()->registerImplIfRequired(opset_factory, NAME, NAMESPACE::NAME::get_type_info_static(), opset.get_default_builder<NAMESPACE::NAME>()); \
                                          opset.insert_type_info_only(NAMESPACE::NAME::get_type_info_static().name, NAMESPACE::NAME::get_type_info_static());
#include "openvino/opsets/opset9_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ngraph::OpSet& ngraph::get_opset1() {
    static OpSet opset(ov::get_opset1());
    return opset;
}

const ngraph::OpSet& ngraph::get_opset2() {
    static OpSet opset(ov::get_opset2());
    return opset;
}

const ngraph::OpSet& ngraph::get_opset3() {
    static OpSet opset(ov::get_opset3());
    return opset;
}

const ngraph::OpSet& ngraph::get_opset4() {
    static OpSet opset(ov::get_opset4());
    return opset;
}

const ngraph::OpSet& ngraph::get_opset5() {
    static OpSet opset(ov::get_opset5());
    return opset;
}

const ngraph::OpSet& ngraph::get_opset6() {
    static OpSet opset(ov::get_opset6());
    return opset;
}

const ngraph::OpSet& ngraph::get_opset7() {
    static OpSet opset(ov::get_opset7());
    return opset;
}

const ngraph::OpSet& ngraph::get_opset8() {
    static OpSet opset(ov::get_opset8());
    return opset;
}

const ngraph::OpSet& ngraph::get_opset9() {
    static OpSet opset(ov::get_opset9());
    return opset;
}
