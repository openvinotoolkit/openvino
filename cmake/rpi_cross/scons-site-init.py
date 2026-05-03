# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import SCons.Platform
import SCons.Script

_environment = SCons.Script.Environment


def Environment(*args, **kwargs):
    kwargs.setdefault("tools", ["gcc", "g++", "gnulink", "ar", "as", "gas"])
    env = _environment(*args, **kwargs)
    env["TEMPFILE"] = SCons.Platform.TempFileMunge
    env["ARCOM"] = "$AR $ARFLAGS $TARGET ${TEMPFILE('$SOURCES')}"
    env.setdefault("RANLIBCOM", "$RANLIB $RANLIBFLAGS $TARGET")
    env.setdefault("RANLIBFLAGS", [])
    return env


SCons.Script.Environment = Environment
