// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The upstream execution_config.cpp implemented ov::PluginConfig finalization
// over ov::Model/ov::AnyMap/remote contexts, all of which were dropped in the
// standalone core. ExecutionConfig is now header-only (see execution_config.hpp);
// this translation unit is kept so the source layout matches the upstream tree.

#include "execution_config.hpp"
