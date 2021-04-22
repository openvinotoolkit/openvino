// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"

#include "ngraph/pass/constant_folding.hpp"
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
            if (pass_name == "ConstantFolding")
                push_pass<ngraph::pass::ConstantFolding>();

            if (m_per_pass_validation)
            {
                push_pass<ngraph::pass::Validate>();
            }
            return;
        }
    };
} // namespace

void regclass_pyngraph_passes_Manager(py::module m)
{
    py::class_<ManagerWrapper> manager(m, "Manager");
    manager.doc() = "ngraph.impl.passes.Manager wraps ngraph::pass::Manager using ManagerWrapper";

    manager.def(py::init<>());

    manager.def("set_per_pass_validation", &ManagerWrapper::set_per_pass_validation);
    manager.def("run_passes", &ManagerWrapper::run_passes);
    manager.def("register_pass",
                (void (ManagerWrapper::*)(std::string)) & ManagerWrapper::register_pass);
}
