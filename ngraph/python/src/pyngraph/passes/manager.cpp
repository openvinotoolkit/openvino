//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"

#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/convert_fp32_to_fp16.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/validate.hpp"

#include "pyngraph/passes/manager.hpp"

namespace py = pybind11;

namespace
{
    class ManagerWrapper : public ngraph::pass::Manager
    {
    public:
        ManagerWrapper() {}
        ~ManagerWrapper() {}
        void register_pass(std::string pass_name)
        {
            if (pass_name == "AlgebraicSimplification")
                push_pass<ngraph::pass::AlgebraicSimplification>();
            if (pass_name == "ConstantFolding")
                push_pass<ngraph::pass::ConstantFolding>();
            if (pass_name == "ConvertFP32ToFP16")
                push_pass<ngraph::pass::ConvertFP32ToFP16>();
            if (pass_name == "GetOutputElementElimination")
                push_pass<ngraph::pass::GetOutputElementElimination>();
            if (pass_name == "NopElimination")
                push_pass<ngraph::pass::NopElimination>();
            if (pass_name == "Validate")
                push_pass<ngraph::pass::Validate>();

            if (m_per_pass_validation)
            {
                push_pass<ngraph::pass::Validate>();
            }
            return;
        }
    };
}

void regclass_pyngraph_passes_Manager(py::module m)
{
    // py::class_<ngraph::pass::Manager, std::shared_ptr<ngraph::pass::Manager>> manager_base(
    //     m, "ManagerBase");
    // py::class_<ManagerWrapper, std::shared_ptr<ManagerWrapper>, ngraph::pass::Manager> manager(
    //     m, "Manager");
    py::class_<ManagerWrapper> manager(m, "Manager");
    manager.doc() = "ngraph.impl.passes.Manager wraps ngraph::pass::Manager using ManagerWrapper";

    manager.def(py::init<>());

    manager.def("set_pass_visualization", &ManagerWrapper::set_pass_visualization);
    manager.def("set_per_pass_validation", &ManagerWrapper::set_per_pass_validation);
    manager.def("run_passes", &ManagerWrapper::run_passes);
    manager.def("register_pass",
                (void (ManagerWrapper::*)(std::string)) & ManagerWrapper::register_pass);
}
