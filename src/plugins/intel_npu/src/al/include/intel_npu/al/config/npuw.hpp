// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>

#include "common.hpp"
#include "npu_private_properties.hpp"
#include "npuw_private_properties.hpp"

namespace intel_npu {

//
// register
//

void registerNPUWOptions(OptionsDesc& desc);

struct NPU_USE_NPUW final : OptionBase<NPU_USE_NPUW, bool> {
    static std::string_view key() {
        return ov::intel_npu::use_npuw.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPU_FROM_NPUW final : OptionBase<NPU_FROM_NPUW, bool> {
    static std::string_view key() {
        return ov::intel_npu::from_npuw.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_DEVICES final : OptionBase<NPUW_DEVICES, std::string> {
    static std::string_view key() {
        return ov::intel_npu::npuw::devices.name();
    }

    static std::string defaultValue() {
        return "NPU,CPU";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_SUBMODEL_DEVICE final : OptionBase<NPUW_SUBMODEL_DEVICE, std::string> {
    static std::string_view key() {
        return ov::intel_npu::npuw::submodel_device.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_NUM_SUBMODELS final : OptionBase<NPUW_NUM_SUBMODELS, std::size_t> {
    static std::string_view key() {
        return ov::intel_npu::npuw::num_submodels.name();
    }

    static std::size_t defaultValue() {
        return {};
    }

    static constexpr std::string_view getTypeName() {
        return "std::size_t";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_ONLINE_PIPELINE final : OptionBase<NPUW_ONLINE_PIPELINE, std::string> {
    static std::string_view key() {
        return ov::intel_npu::npuw::partitioning::online::pipeline.name();
    }

    static std::string defaultValue() {
        return "REP";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_ONLINE_AVOID final : OptionBase<NPUW_ONLINE_AVOID, std::string> {
    static std::string_view key() {
        return ov::intel_npu::npuw::partitioning::online::avoid.name();
    }

    static std::string defaultValue() {
        return "";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_ONLINE_MIN_SIZE final : OptionBase<NPUW_ONLINE_MIN_SIZE, std::size_t> {
    static std::string_view key() {
        return ov::intel_npu::npuw::partitioning::online::min_size.name();
    }

    static std::size_t defaultValue() {
        return 10;
    }

    static constexpr std::string_view getTypeName() {
        return "std::size_t";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_ONLINE_DUMP_PLAN final : OptionBase<NPUW_ONLINE_DUMP_PLAN, std::string> {
    static std::string_view key() {
        return ov::intel_npu::npuw::partitioning::online::dump_plan.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_PLAN final : OptionBase<NPUW_PLAN, std::string> {
    static std::string_view key() {
        return ov::intel_npu::npuw::partitioning::plan.name();
    }

    static std::string defaultValue() {
        return "";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_FOLD final : OptionBase<NPUW_FOLD, bool> {
    static std::string_view key() {
        return ov::intel_npu::npuw::partitioning::fold.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_CWAI final : OptionBase<NPUW_CWAI, bool> {
    static std::string_view key() {
        return ov::intel_npu::npuw::partitioning::cwai.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_DCOFF_TYPE final : OptionBase<NPUW_DCOFF_TYPE, std::string> {
    static std::string_view key() {
        return ov::intel_npu::npuw::partitioning::dcoff_type.name();
    }

    static std::string defaultValue() {
        return "";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_DCOFF_SCALE final : OptionBase<NPUW_DCOFF_SCALE, bool> {
    static std::string_view key() {
        return ov::intel_npu::npuw::partitioning::dcoff_with_scale.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_FUNCALL_FOR_ALL final : OptionBase<NPUW_FUNCALL_FOR_ALL, bool> {
    static std::string_view key() {
        return ov::intel_npu::npuw::partitioning::funcall_for_all.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_PARALLEL_COMPILE final : OptionBase<NPUW_PARALLEL_COMPILE, bool> {
    static std::string_view key() {
        return ov::intel_npu::npuw::parallel_compilation.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_FUNCALL_ASYNC final : OptionBase<NPUW_FUNCALL_ASYNC, bool> {
    static std::string_view key() {
        return ov::intel_npu::npuw::funcall_async.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_ACC_CHECK final : OptionBase<NPUW_ACC_CHECK, bool> {
    static std::string_view key() {
        return ov::intel_npu::npuw::accuracy::check.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_ACC_THRESH final : OptionBase<NPUW_ACC_THRESH, double> {
    static std::string_view key() {
        return ov::intel_npu::npuw::accuracy::threshold.name();
    }

    static double defaultValue() {
        return 0.01;
    }

    static constexpr std::string_view getTypeName() {
        return "double";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_ACC_DEVICE final : OptionBase<NPUW_ACC_DEVICE, std::string> {
    static std::string_view key() {
        return ov::intel_npu::npuw::accuracy::reference_device.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_DUMP_FULL final : OptionBase<NPUW_DUMP_FULL, bool> {
    static std::string_view key() {
        return ov::intel_npu::npuw::dump::full.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_DUMP_SUBS final : OptionBase<NPUW_DUMP_SUBS, std::string> {
    static std::string_view key() {
        return ov::intel_npu::npuw::dump::subgraph.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_DUMP_SUBS_ON_FAIL final : OptionBase<NPUW_DUMP_SUBS_ON_FAIL, std::string> {
    static std::string_view key() {
        return ov::intel_npu::npuw::dump::subgraph_on_fail.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_DUMP_IO final : OptionBase<NPUW_DUMP_IO, std::string> {
    static std::string_view key() {
        return ov::intel_npu::npuw::dump::inputs_outputs.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_DUMP_IO_ITERS final : OptionBase<NPUW_DUMP_IO_ITERS, bool> {
    static std::string_view key() {
        return ov::intel_npu::npuw::dump::io_iters.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};
}  // namespace intel_npu
