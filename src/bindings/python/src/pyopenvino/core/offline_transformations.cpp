// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/offline_transformations.hpp"

#include <pybind11/stl.h>

#include <compress_quantize_weights.hpp>
#include <generate_mapping_file.hpp>
#include <openvino/pass/make_stateful.hpp>
#include <openvino/pass/serialize.hpp>
#include <pot_transformations.hpp>
#include <pruning.hpp>
#include <transformations/common_optimizations/compress_float_constants.hpp>
#include <transformations/common_optimizations/mark_precision_sensitive_subgraphs.hpp>
#include <transformations/common_optimizations/moc_legacy_transformations.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/serialize.hpp>

#include "openvino/pass/low_latency.hpp"
#include "openvino/pass/manager.hpp"

using Version = ov::pass::Serialize::Version;

inline Version convert_to_version(const std::string& version) {
    if (version == "UNSPECIFIED")
        return Version::UNSPECIFIED;
    if (version == "IR_V10")
        return Version::IR_V10;
    if (version == "IR_V11")
        return Version::IR_V11;
    throw ov::Exception("Invoked with wrong version argument: '" + version +
                        "'! The supported versions are: 'UNSPECIFIED'(default), 'IR_V10', 'IR_V11'.");
}

namespace py = pybind11;

void regmodule_offline_transformations(py::module m) {
    py::module m_offline_transformations = m.def_submodule("offline_transformations", "Offline transformations module");

    m_offline_transformations.def(
        "apply_moc_transformations",
        [](std::shared_ptr<ov::Model> function, bool cf) {
            ov::pass::Manager manager;
            manager.register_pass<ngraph::pass::MOCTransformations>(cf);
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("cf"));

    m_offline_transformations.def(
        "apply_moc_legacy_transformations",
        [](std::shared_ptr<ov::Model> function, const std::vector<std::string>& params_with_custom_types) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::MOCLegacyTransformations>(params_with_custom_types);
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("params_with_custom_types"));

    m_offline_transformations.def(
        "apply_pot_transformations",
        [](std::shared_ptr<ov::Model> function, std::string device) {
            ov::pass::Manager manager;
            manager.register_pass<ngraph::pass::POTTransformations>(std::move(device));
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("device"));

    m_offline_transformations.def(
        "apply_low_latency_transformation",
        [](std::shared_ptr<ov::Model> function, bool use_const_initializer = true) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::LowLatency2>(use_const_initializer);
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("use_const_initializer") = true);

    m_offline_transformations.def(
        "apply_pruning_transformation",
        [](std::shared_ptr<ov::Model> function) {
            ov::pass::Manager manager;
            manager.register_pass<ngraph::pass::Pruning>();
            manager.run_passes(function);
        },
        py::arg("function"));

    m_offline_transformations.def(
        "generate_mapping_file",
        [](std::shared_ptr<ov::Model> function, std::string path, bool extract_names) {
            ov::pass::Manager manager;
            manager.register_pass<ngraph::pass::GenerateMappingFile>(path, extract_names);
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("path"),
        py::arg("extract_names"));

    m_offline_transformations.def(
        "apply_make_stateful_transformation",
        [](std::shared_ptr<ov::Model> function, const std::map<std::string, std::string>& param_res_names) {
            ngraph::pass::Manager manager;
            manager.register_pass<ov::pass::MakeStateful>(param_res_names);
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("param_res_names"));

    m_offline_transformations.def(
        "compress_model_transformation",
        [](std::shared_ptr<ov::Model> function) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::MarkPrecisionSensitiveSubgraphs>();
            manager.register_pass<ov::pass::CompressFloatConstants>();
            manager.run_passes(function);
        },
        py::arg("function"));

    m_offline_transformations.def(
        "compress_quantize_weights_transformation",
        [](std::shared_ptr<ov::Model> function) {
            ov::pass::Manager manager;
            manager.register_pass<ngraph::pass::CompressQuantizeWeights>();
            manager.register_pass<ngraph::pass::ZeroPointOptimizer>();
            manager.run_passes(function);
        },
        py::arg("function"));

    // todo: remove as serialize as part of passManager api will be merged
    m_offline_transformations.def(
        "serialize",
        [](std::shared_ptr<ov::Model> function,
           const std::string& path_to_xml,
           const std::string& path_to_bin,
           const std::string& version) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::Serialize>(path_to_xml, path_to_bin, convert_to_version(version));
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("model_path"),
        py::arg("weights_path"),
        py::arg("version") = "UNSPECIFIED",
        R"(
    Serialize given function into IR. The generated .xml and .bin files will be save
    into provided paths.
    Parameters
    ----------
    function : ov.Model
        function which will be converted to IR representation
    xml_path : str
        path where .xml file will be saved
    bin_path : str
        path where .bin file will be saved
    version : str
        sets the version of the IR which will be generated.
        Supported versions are:
                        - "UNSPECIFIED" (default) : Use the latest or function version
                        - "IR_V10" : v10 IR
                        - "IR_V11" : v11 IR

    Examples:
    ----------
    1. Default IR version:
        shape = [2, 2]
        parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
        parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
        parameter_c = ov.parameter(shape, dtype=np.float32, name="C")
        model = (parameter_a + parameter_b) * parameter_c
        func = Model(model, [parameter_a, parameter_b, parameter_c], "Model")
        # IR generated with default version 
        serialize(func, model_path="./serialized.xml", weights_path="./serialized.bin")

    2. IR version 11:
        shape = [2, 2]
        parameter_a = ov.parameter(shape, dtype=np.float32, name="A")
        parameter_b = ov.parameter(shape, dtype=np.float32, name="B")
        parameter_c = ov.parameter(shape, dtype=np.float32, name="C")
        model = (parameter_a + parameter_b) * parameter_c
        func = Model(model, [parameter_a, parameter_b, parameter_c], "Model")
        # IR generated with default version 
        serialize(func, model_path="./serialized.xml", "./serialized.bin", version="IR_V11")    
    // )");
}
