// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file itt.hpp
 */

#pragma once

#include <openvino/cc/factory.h>
#include <openvino/cc/selective_build.h>

#include <openvino/cc/pass/itt.hpp>
#include <openvino/itt.hpp>

namespace ov {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(core);
OV_ITT_DOMAIN(ov_pass);
OV_ITT_DOMAIN(ov_op, "ov::Op");
}  // namespace domains
}  // namespace itt
}  // namespace ov
OV_CC_DOMAINS(ov_op);
OV_CC_DOMAINS(ov_opset);

/*
 * REGISTER_OP registers Operation creation inside OpSet
 * INSERT_OP macro allows to ignore some Operations inside OpSet if it's creation wasn't registered
 */
#if defined(SELECTIVE_BUILD_ANALYZER)
#    define OV_OP_SCOPE(region) OV_SCOPE(ov_op, region)
#    define REGISTER_OP(opset_name, op_name) \
        OV_ITT_SCOPED_TASK(SIMPLE_ov_opset, openvino::itt::handle(opset_name + "_" + op_name))
#    define INSERT_OP(opset_name, op_name, op_namespace) opset.insert<op_namespace::op_name>()
#elif defined(SELECTIVE_BUILD)
#    define OV_OP_SCOPE(region)                                        \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_op, _, region)) == 0) \
        OPENVINO_THROW(std::string(OV_PP_TOSTRING(OV_PP_CAT3(ov_op, _, region))) + " is disabled!")
#    define REGISTER_OP(opset_name, op_name)
#    define INSERT_OP(opset_name, op_name, op_namespace)                                \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT4(ov_opset_, opset_name, _, op_name)) == 1) \
        opset.insert<op_namespace::op_name>()
#else
#    define OV_OP_SCOPE(region) OV_ITT_SCOPED_TASK(ov::itt::domains::ov_op, OV_PP_TOSTRING(region))
#    define REGISTER_OP(opset_name, op_name)
#    define INSERT_OP(opset_name, op_name, op_namespace) opset.insert<op_namespace::op_name>()
#endif

#define OPENVINO_TYPE_CASE(region, a, ...)                                                                \
    case ov::element::Type_t::a: {                                                                        \
        OV_SCOPE(ov_op, OV_PP_CAT3(region, _, a)) { rc = evaluate<ov::element::Type_t::a>(__VA_ARGS__); } \
    } break

#define OPENVINO_2_TYPES_CASE(region, a, b, ...)                                \
    case element::Type_t::a: {                                                  \
        OV_SCOPE(ov_op, OV_PP_CAT4(region, _, a, b)) {                          \
            rc = evaluate<element::Type_t::a, element::Type_t::b>(__VA_ARGS__); \
        }                                                                       \
    } break

#define OPENVINO_COPY_TENSOR(region, a, ...)                                                                 \
    case ov::element::Type_t::a: {                                                                           \
        OV_SCOPE(ov_op, OV_PP_CAT3(region, _, a)) { rc = copy_tensor<ov::element::Type_t::a>(__VA_ARGS__); } \
    } break
