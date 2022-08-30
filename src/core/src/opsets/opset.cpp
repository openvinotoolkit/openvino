// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"

#include "ngraph/log.hpp"
#include "ngraph/ops.hpp"

ngraph::OpSet::OpSet(const ov::OpSet& opset) : ov::OpSet(opset) {}

std::mutex& ov::OpSet::get_mutex() {
    static std::mutex opset_mutex;
    return opset_mutex;
}

ov::Node* ov::OpSet::create(const std::string& name) const {
    auto type_info_it = m_name_type_info_map.find(name);
    if (type_info_it == m_name_type_info_map.end()) {
        NGRAPH_WARN << "Couldn't create operator of type: " << name << " . Operation not registered in opset.";
        return nullptr;
    }
    return m_factory_registry.create(type_info_it->second);
}

ov::Node* ov::OpSet::create_insensitive(const std::string& name) const {
    auto type_info_it = m_case_insensitive_type_info_map.find(to_upper_name(name));
    return type_info_it == m_case_insensitive_type_info_map.end() ? nullptr
                                                                  : m_factory_registry.create(type_info_it->second);
}

const ov::OpSet& ov::get_opset1() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset1_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset2() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset2_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset3() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset3_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset4() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset4_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset5() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset5_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset6() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset6_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset7() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset7_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset8() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset8_tbl.hpp"
#undef _OPENVINO_OP_REG
    });
    return opset;
}

const ov::OpSet& ov::get_opset9() {
    static OpSet opset;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
#define _OPENVINO_OP_REG(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
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
