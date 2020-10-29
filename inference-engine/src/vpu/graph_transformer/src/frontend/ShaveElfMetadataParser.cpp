// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/frontend/ShaveElfMetadataParser.h"
#include <algorithm>

namespace {

// two operand operator evaluation
uint32_t md_eval_expression_type_op_2(
    const md_expr_node_type_t type,
    const uint32_t lhs,
    const uint32_t rhs) {
  switch (type) {
  case md_type_op_umul:    return lhs * rhs;
  case md_type_op_udiv:    return lhs / rhs;
  case md_type_op_add:     return (int32_t)lhs + (int32_t)rhs;
  case md_type_op_sub:     return (int32_t)lhs - (int32_t)rhs;
  case md_type_op_min:     return std::min((int32_t)lhs, (int32_t)rhs);
  case md_type_op_max:     return std::max((int32_t)lhs, (int32_t)rhs);
  case md_type_op_umin:    return std::min(lhs, rhs);
  case md_type_op_umax:    return std::max(lhs, rhs);
  case md_type_op_and:     return lhs & rhs;
  case md_type_op_or:      return lhs | rhs;
  case md_type_op_xor:     return lhs ^ rhs;
  case md_type_op_shl:     return lhs << rhs;
  case md_type_op_lshr:    return lhs >> rhs;
  default:
    assert(!"unknown node type");
    return 0;
  }
}
} // namespace

uint32_t md_parser_t::evaluate_expr(const md_expr_t *expression,
                                    const uint32_t local_size[3],
                                    const uint32_t global_size[3],
                                    const uint32_t *param,
                                    uint32_t param_count) const {
  // find the nodes for the given expr_index
  assert(expression->node_first < hdr->expr_node_count);
  const md_expr_node_t *node = expr_node + expression->node_first;
  // the intermediate value stack
  std::vector<uint32_t> values;
  // for all of the nodes in this expression
  for (uint32_t i = 0; i < expression->node_count; ++i) {
    // get the node
    const md_expr_node_t &v = node[i];
    // dispatch the opcode
    switch (v.type) {
    case md_type_immediate:
      values.push_back(v.value);
      break;
    case md_type_op_umul: {
    case md_type_op_udiv:
    case md_type_op_add:
    case md_type_op_sub:
    case md_type_op_min:
    case md_type_op_max:
    case md_type_op_umin:
    case md_type_op_umax:
    case md_type_op_and:
    case md_type_op_or:
    case md_type_op_xor:
    case md_type_op_shl:
    case md_type_op_lshr:
      uint32_t rhs = values.rbegin()[0];
      uint32_t lhs = values.rbegin()[1];
      values.pop_back();
      values.back() = md_eval_expression_type_op_2(v.type, lhs, rhs);
    }
      break;
    case md_type_global_size:
      assert(v.value < 3);
      values.push_back(global_size[v.value]);
      break;
    case md_type_local_size:
      assert(v.value < 3);
      values.push_back(local_size[v.value]);
      break;
    case md_type_param:
      assert(v.value < param_count);
      values.push_back(param[v.value]);
      break;
    default:
      assert(!"unknown node type");
    }
  }
  // should only be one value remaining which is the result
  assert(values.size() == 1);
  return values.back();
}
