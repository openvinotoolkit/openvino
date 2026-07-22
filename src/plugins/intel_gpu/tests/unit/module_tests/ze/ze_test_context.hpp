// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef OV_GPU_WITH_ZE_RT

#include "test_utils.h"

#include <exception>
#include <memory>

namespace ze_tests {

struct ze_test_context {
	std::shared_ptr<cldnn::engine> ze_test_engine;
	std::shared_ptr<cldnn::stream> ze_test_stream;
};

inline ze_test_context create_ze_test_context() {
	ze_test_context ctx;
	try {
		ctx.ze_test_engine = ::tests::create_test_engine(cldnn::engine_types::ze, cldnn::runtime_types::ze);
	} catch (const std::exception& e) {
		OPENVINO_THROW(e.what());
	}

	OPENVINO_ASSERT(ctx.ze_test_engine != nullptr, "[GPU] Failed to create ZE engine for tests");
	ctx.ze_test_stream = ctx.ze_test_engine->create_stream(::tests::get_test_default_config(*ctx.ze_test_engine));
	return ctx;
}

}  // namespace ze_tests

#endif  // OV_GPU_WITH_ZE_RT
