// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(OV_CPU_WITH_ACL)
#    if defined(OPENVINO_ARCH_ARM)
#        define OV_CPU_INSTANCE_ACL32(...) {__VA_ARGS__},
#        define OV_CPU_ACL32(...)          __VA_ARGS__
#    else
#        define OV_CPU_INSTANCE_ACL32(...)
#        define OV_CPU_ACL32(...)
#    endif
#    if defined(OPENVINO_ARCH_ARM64)
#        define OV_CPU_INSTANCE_ACL64(...) {__VA_ARGS__},
#        define OV_CPU_ACL64(...)          __VA_ARGS__
#    else
#        define OV_CPU_INSTANCE_ACL64(...)
#        define OV_CPU_ACL64(...)
#    endif
#    if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
#        define OV_CPU_INSTANCE_ACL(...) {__VA_ARGS__},
#        define OV_CPU_ACL(...)          __VA_ARGS__
#    else
#        define OV_CPU_INSTANCE_ACL(...)
#        define OV_CPU_ACL(...)
#    endif
#else
#    define OV_CPU_INSTANCE_ACL32(...)
#    define OV_CPU_ACL32(...)
#    define OV_CPU_INSTANCE_ACL64(...)
#    define OV_CPU_ACL64(...)
#    define OV_CPU_INSTANCE_ACL(...)
#    define OV_CPU_ACL(...)
#endif

#if defined(OV_CPU_WITH_DNNL)
#    define OV_CPU_INSTANCE_DNNL(...) {__VA_ARGS__},
#    define OV_CPU_DNNL(...)          __VA_ARGS__
#else
#    define OV_CPU_INSTANCE_DNNL(...)
#    define OV_CPU_DNNL(...)
#endif

#if defined(OV_CPU_WITH_KLEIDIAI)
#    define OV_CPU_INSTANCE_KLEIDIAI(...) {__VA_ARGS__},
#    define OV_CPU_KLEIDIAI(...)          __VA_ARGS__
#else
#    define OV_CPU_INSTANCE_KLEIDIAI(...)
#    define OV_CPU_KLEIDIAI(...)
#endif

#if defined(OV_CPU_WITH_DNNL) && defined(OPENVINO_ARCH_X86_64)
#    define OV_CPU_INSTANCE_DNNL_X64(...) {__VA_ARGS__},
#    define OV_CPU_DNNL_X64(...)          __VA_ARGS__
#else
#    define OV_CPU_INSTANCE_DNNL_X64(...)
#    define OV_CPU_DNNL_X64(...)
#endif

#if defined(OPENVINO_ARCH_ARM64)
#    define OV_CPU_INSTANCE_ARM64(...) {__VA_ARGS__},
#    define OV_CPU_ARM64(...)          __VA_ARGS__
#else
#    define OV_CPU_INSTANCE_ARM64(...)
#    define OV_CPU_ARM64(...)
#endif

#if defined(OV_CPU_WITH_DNNL) && defined(OPENVINO_ARCH_ARM64)
#    define OV_CPU_INSTANCE_DNNL_ARM64(...) {__VA_ARGS__},
#    define OV_CPU_DNNL_ARM64(...)          __VA_ARGS__
#else
#    define OV_CPU_INSTANCE_DNNL_ARM64(...)
#    define OV_CPU_DNNL_ARM64(...)
#endif

#if defined(OPENVINO_ARCH_X86_64)
#    define OV_CPU_INSTANCE_X64(...) {__VA_ARGS__},
#    define OV_CPU_X64(...)          __VA_ARGS__
#else
#    define OV_CPU_INSTANCE_X64(...)
#    define OV_CPU_X64(...)
#endif

#if defined(OV_CPU_WITH_MLAS) && defined(OPENVINO_ARCH_ARM64)
#    define OV_CPU_INSTANCE_MLAS_ARM64(...) {__VA_ARGS__},
#    define OV_CPU_MLAS_ARM64(...)          __VA_ARGS__
#else
#    define OV_CPU_INSTANCE_MLAS_ARM64(...)
#    define OV_CPU_MLAS_ARM64(...)
#endif

#if defined(OV_CPU_WITH_MLAS) && defined(OPENVINO_ARCH_X86_64)
#    define OV_CPU_INSTANCE_MLAS_X64(...) {__VA_ARGS__},
#    define OV_CPU_MLAS_X64(...)          __VA_ARGS__
#else
#    define OV_CPU_INSTANCE_MLAS_X64(...)
#    define OV_CPU_MLAS_X64(...)
#endif

#if defined(OV_CPU_WITH_SHL)
#    define OV_CPU_INSTANCE_SHL(...) {__VA_ARGS__},
#    define OV_CPU_SHL(...)          __VA_ARGS__
#else
#    define OV_CPU_INSTANCE_SHL(...)
#    define OV_CPU_SHL(...)
#endif

#if defined(OPENVINO_ARCH_RISCV64)
#    define OV_CPU_INSTANCE_RISCV64(...) {__VA_ARGS__},
#    define OV_CPU_RISCV64(...)          __VA_ARGS__
#else
#    define OV_CPU_INSTANCE_RISCV64(...)
#    define OV_CPU_RISCV64(...)
#endif

#define OV_CPU_INSTANCE_COMMON(...) {__VA_ARGS__},
#define OV_CPU_COMMON(...)          __VA_ARGS__
