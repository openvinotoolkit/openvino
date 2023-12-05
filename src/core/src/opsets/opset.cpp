// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"

#include "itt.hpp"
#include "ngraph/log.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/util/log.hpp"

ov::OpSet::OpSet(const std::string& name) : m_name(name) {}

ov::OpSet::OpSet(const ov::OpSet& opset) {
    *this = opset;
}

ov::OpSet& ov::OpSet::operator=(const ov::OpSet& opset) {
    m_factory_registry = opset.m_factory_registry;
    m_name = opset.m_name;
    m_op_types = opset.m_op_types;
    m_name_type_info_map = opset.m_name_type_info_map;
    m_case_insensitive_type_info_map = opset.m_case_insensitive_type_info_map;
    return *this;
}

ov::Node* ov::OpSet::create(const std::string& name) const {
    auto type_info_it = m_name_type_info_map.find(name);
    if (type_info_it == m_name_type_info_map.end()) {
        OPENVINO_WARN << "Couldn't create operator of type: " << name << " . Operation not registered in opset.";
        return nullptr;
    }
    REGISTER_OP(m_name, name);
    return m_factory_registry.create(type_info_it->second);
}

ov::Node* ov::OpSet::create_insensitive(const std::string& name) const {
    auto type_info_it = m_case_insensitive_type_info_map.find(to_upper_name(name));
    if (type_info_it == m_case_insensitive_type_info_map.end()) {
        OPENVINO_WARN << "Couldn't create operator of type: " << name << " . Operation not registered in opset.";
        return nullptr;
    }
    REGISTER_OP(m_name, name);
    return m_factory_registry.create(type_info_it->second);
}

const std::map<std::string, std::function<const ov::OpSet&()>>& ov::get_available_opsets() {
#define _OPENVINO_REG_OPSET(OPSET) \
    { #OPSET, ov::get_##OPSET }
    const static std::map<std::string, std::function<const ov::OpSet&()>> opset_map = {_OPENVINO_REG_OPSET(opset1),
                                                                                       _OPENVINO_REG_OPSET(opset2),
                                                                                       _OPENVINO_REG_OPSET(opset3),
                                                                                       _OPENVINO_REG_OPSET(opset4),
                                                                                       _OPENVINO_REG_OPSET(opset5),
                                                                                       _OPENVINO_REG_OPSET(opset6),
                                                                                       _OPENVINO_REG_OPSET(opset7),
                                                                                       _OPENVINO_REG_OPSET(opset8),
                                                                                       _OPENVINO_REG_OPSET(opset9),
                                                                                       _OPENVINO_REG_OPSET(opset10),
                                                                                       _OPENVINO_REG_OPSET(opset11),
                                                                                       _OPENVINO_REG_OPSET(opset12),
                                                                                       _OPENVINO_REG_OPSET(opset13)};
#undef _OPENVINO_REG_OPSET
    return opset_map;
}

const ov::OpSet& ov::get_opset1() {
    static OpSet opset("opset1");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) INSERT_OP(opset1, NAME, NAMESPACE);
#include "openvino/opsets/opset1_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset2() {
    static OpSet opset("opset2");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) INSERT_OP(opset2, NAME, NAMESPACE);
#include "openvino/opsets/opset2_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset3() {
    static OpSet opset("opset3");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) INSERT_OP(opset3, NAME, NAMESPACE);
#include "openvino/opsets/opset3_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset4() {
    static OpSet opset("opset4");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) INSERT_OP(opset4, NAME, NAMESPACE);
#include "openvino/opsets/opset4_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset5() {
    static OpSet opset("opset5");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) INSERT_OP(opset5, NAME, NAMESPACE);
#include "openvino/opsets/opset5_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset6() {
    static OpSet opset("opset6");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) INSERT_OP(opset6, NAME, NAMESPACE);
#include "openvino/opsets/opset6_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset7() {
    static OpSet opset("opset7");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) INSERT_OP(opset7, NAME, NAMESPACE);
#include "openvino/opsets/opset7_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset8() {
    static OpSet opset("opset8");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) INSERT_OP(opset8, NAME, NAMESPACE);
#include "openvino/opsets/opset8_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset9() {
    static OpSet opset("opset9");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) INSERT_OP(opset9, NAME, NAMESPACE);
#include "openvino/opsets/opset9_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset10() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset10_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset11() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset11_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset12() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset12_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset13() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset13_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {

OpSet::OpSet(const ov::OpSet& opset) : ov::OpSet(opset) {}

OpSet::OpSet(const OpSet& opset) : ov::OpSet(opset) {}

const std::map<std::string, std::function<const OpSet&()>>& get_available_opsets() {
#define _REG_OPSET(OPSET) \
    { #OPSET, get_##OPSET }
    const static std::map<std::string, std::function<const OpSet&()>> opset_map = {_REG_OPSET(opset1),
                                                                                   _REG_OPSET(opset2),
                                                                                   _REG_OPSET(opset3),
                                                                                   _REG_OPSET(opset4),
                                                                                   _REG_OPSET(opset5),
                                                                                   _REG_OPSET(opset6),
                                                                                   _REG_OPSET(opset7),
                                                                                   _REG_OPSET(opset8),
                                                                                   _REG_OPSET(opset9),
                                                                                   _REG_OPSET(opset10),
                                                                                   _REG_OPSET(opset11),
                                                                                   _REG_OPSET(opset12),
                                                                                   _REG_OPSET(opset13)};
#undef _REG_OPSET
    return opset_map;
}

const OpSet& get_opset1() {
    static OpSet opset(ov::get_opset1());
    return opset;
}

const OpSet& get_opset2() {
    static OpSet opset(ov::get_opset2());
    return opset;
}

const OpSet& get_opset3() {
    static OpSet opset(ov::get_opset3());
    return opset;
}

const OpSet& get_opset4() {
    static OpSet opset(ov::get_opset4());
    return opset;
}

const OpSet& get_opset5() {
    static OpSet opset(ov::get_opset5());
    return opset;
}

const OpSet& get_opset6() {
    static OpSet opset(ov::get_opset6());
    return opset;
}

const OpSet& get_opset7() {
    static OpSet opset(ov::get_opset7());
    return opset;
}

const OpSet& get_opset8() {
    static OpSet opset(ov::get_opset8());
    return opset;
}

const OpSet& get_opset9() {
    static OpSet opset(ov::get_opset9());
    return opset;
}

const OpSet& get_opset10() {
    static OpSet opset(ov::get_opset10());
    return opset;
}

const OpSet& get_opset11() {
    static OpSet opset(ov::get_opset11());
    return opset;
}

const OpSet& get_opset12() {
    static OpSet opset(ov::get_opset12());
    return opset;
}

const OpSet& get_opset13() {
    static OpSet opset(ov::get_opset13());
    return opset;
}

}  // namespace ngraph
