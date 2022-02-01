// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SHAVE_METADATA_PARSER_H_INCLUDED
#define SHAVE_METADATA_PARSER_H_INCLUDED

#include <string>
#include <vector>
#include <cassert>
#include <cstring>

#include "ShaveElfMetadata.h"


struct md_parser_t {
  md_parser_t(const uint8_t *data, size_t data_size,
              const char *strtab,
              size_t strtab_size)
      : hdr(reinterpret_cast<const md_header_t *>(data)),
        kernel_descriptor(reinterpret_cast<const md_kernel_descriptor_t *>(
            data + hdr->kernel_first)),
        kernel_argument(reinterpret_cast<const md_kernel_argument_t *>(
            data + hdr->arg_first)),
        kernel_sipp_info(reinterpret_cast<const md_kernel_sipp_info_t *>(
            data + hdr->sipp_info_first)),
        expr_node(reinterpret_cast<const md_expr_node_t *>(
            data + hdr->expr_node_first)),
        expr(reinterpret_cast<const md_expr_t *>(data + hdr->expr_first)),
        func(reinterpret_cast<const md_function_t *>(data + hdr->func_first)),
        strtab(strtab), strtab_size(strtab_size) {
    (void)data_size;
    (void)strtab_size;
    assert(hdr->version == md_version_latest);
  }

  // Return the metadata version
  //
  md_version_t get_version() const {
    return static_cast<md_version_t>(hdr->version);
  }

  // Get a kernel by name
  //
  const md_kernel_descriptor_t *get_kernel(const std::string &name) const {
    for (uint32_t i=0; i < hdr->kernel_count; ++i) {
      const md_kernel_descriptor_t *d = get_kernel(i);
      const char *n = get_name(d);
      if (name == n) {
        return d;
      }
    }
    return nullptr;
  }

  // Get a kernel id by name
  //
  int get_kernel_id(const std::string& name) const {
      for (uint32_t i = 0; i < hdr->kernel_count; ++i) {
          const md_kernel_descriptor_t* d = get_kernel(i);
          const char* n = get_name(d);
          if (name == n) {
              return i;
          }
      }
      return -1;
  }

  // Return true if a kernel has a specific variant
  //
  bool kernel_has_variant(const md_kernel_descriptor_t *kernel,
                          md_kernel_variant_type_t variant) const {
    const auto &v = kernel->variant[ variant ];
    return v.name != md_invalid_index &&
           v.func != md_invalid_index;
  }

  // return the load address of a kernel variant
  //
  uint32_t get_kernel_load_addr(const md_kernel_descriptor_t *kernel, const md_kernel_variant_type_t variant) {
    if (!kernel_has_variant(kernel, variant)) {
      return 0;
    }
    const auto &v = kernel->variant[ variant ];
    const md_function_t &f = func[v.func];
    return f.load_address;
  }

  // Get a rough stack size estimate for a kernel variant
  //
  uint32_t get_kernel_stack_estimate(const md_kernel_descriptor_t *kernel,
                                     md_kernel_variant_type_t variant,
                                     const uint32_t local_size[3]) const {
    const uint32_t local_area = local_size[0] * local_size[1] * local_size[2];
    const uint32_t per_wi = local_area * kernel->stack_size_wi;
    const uint32_t per_wg = kernel->stack_size_wg;
    const uint32_t factor = kernel->variant[variant].factor;
    switch (variant) {
    case md_variant_vectorized:
    case md_variant_unrolled:     return per_wg + per_wi * factor;
    case md_variant_scalar:
    default:                      return per_wg + per_wi;
    }
  }

  // Return the number of local arguments a kernel has
  //
  uint32_t get_num_local_args(const md_kernel_descriptor_t *kernel) const {
    uint32_t out = 0;
    for (uint32_t i = 0; i < kernel->arg_count; ++i) {
      const md_kernel_argument_t *arg = get_argument(kernel->arg_index + i);
      out += arg->addr_space == md_addr_space_local;
    }
    return out;
  }

  // Get the number of distinct kernels in this file
  //
  uint32_t get_kernel_count() const {
    return hdr->kernel_count;
  }

  // Get a function by index
  //
  const md_function_t *get_func_ptr(uint32_t index) const {
    assert(index != md_invalid_index && index < hdr->func_count);
    return func + index;
  }

  // Get a kernel by load address
  //
  const md_kernel_descriptor_t *get_kernel_by_addr(uint32_t addr) const {
    for (uint32_t i = 0; i < hdr->kernel_count; ++i) {
      const md_kernel_descriptor_t *desc = get_kernel(i);
      for (uint32_t j = 0; j < md_VARIANT_COUNT; ++j) {
        const uint32_t index = desc->variant[j].func;
        if (index == md_invalid_index) {
          continue;
        }
        const md_function_t *ptr = get_func_ptr(index);
        if (ptr->load_address == addr) {
          return desc;
        }
      }
    }
    return nullptr;
  }

  // Get a kernel by index
  //
  const md_kernel_descriptor_t *get_kernel(uint32_t index) const {
    assert(index < hdr->kernel_count);
    return kernel_descriptor + index;
  }

  // Get an argument by index
  //
  const md_kernel_argument_t *get_argument(uint32_t index) const {
    assert(index < hdr->arg_count);
    return kernel_argument + index;
  }

  // Get SIPP info by index
  //
  const md_kernel_sipp_info_t *get_sipp_info(uint32_t index) const {
    assert(index < hdr->sipp_info_count);
    return kernel_sipp_info + index;
  }

  // Get an expression node by index
  //
  const md_expr_node_t *get_expr_node(uint32_t index) const {
    assert(index < hdr->expr_node_count);
    return expr_node + index;
  }

  // Get an expression by index
  //
  const md_expr_t *get_expr(uint32_t index) const {
    assert(index < hdr->expr_count);
    return expr + index;
  }

  // Get a kernel argument for a specific kernel by position
  //
  const md_kernel_argument_t *get_argument(const md_kernel_descriptor_t *kernel, uint32_t index) const {
    assert(index < kernel->arg_count);
    return get_argument(kernel->arg_index + index);
  }

  // Return the name of a kernel
  //
  const char *get_name(const md_kernel_descriptor_t *kernel) const {
    return strtab + kernel->name;
  }

  // Return the name of an argument
  //
  const char *get_name(const md_kernel_argument_t *arg) const {
    return strtab + arg->name;
  }

  // Evaluate an arbitary expression
  //
  uint32_t evaluate_expr(const md_expr_t *expression,
                         const uint32_t local_size[3],
                         const uint32_t global_size[3],
                         const uint32_t *param,
                         uint32_t param_count) const;

protected:
  // structure parsers
  const md_header_t *hdr;
  const md_kernel_descriptor_t *kernel_descriptor;
  const md_kernel_argument_t *kernel_argument;
  const md_kernel_sipp_info_t *kernel_sipp_info;
  const md_expr_node_t *expr_node;
  const md_expr_t *expr;
  const md_function_t *func;
  // string table
  const char *strtab;
  const size_t strtab_size;
};

#endif  // SHAVE_METADATA_PARSER_H_INCLUDED
