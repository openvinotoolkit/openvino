#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Little-endian byte encoders for the bytecode artifact format."""

from __future__ import annotations


def _u8(value: int) -> bytes:
    return int(value).to_bytes(1, "little", signed=False)


def _u16(value: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=False)


def _u64(value: int) -> bytes:
    return int(value).to_bytes(8, "little", signed=False)


def _i16(value: int) -> bytes:
    if value < -32768 or value > 32767:
        raise ValueError(f"int16 operand out of range: {value}")
    return int(value).to_bytes(2, "little", signed=True)


def _i64(value: int) -> bytes:
    if value < -(1 << 63) or value > ((1 << 63) - 1):
        raise ValueError(f"int64 immediate out of range: {value}")
    return int(value).to_bytes(8, "little", signed=True)
