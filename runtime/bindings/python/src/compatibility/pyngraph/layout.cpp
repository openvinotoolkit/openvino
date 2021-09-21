#include "openvino/core/layout.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "ngraph/type.hpp"
#include "pyngraph/layout.hpp"

namespace py = pybind11;



void regclass_pyngraph_Layout(py::module m) {
    py::class_<ov::Layout, std::shared_ptr<ov::Layout>> layout(m, "Layout");
    layout.doc() = "ngraph.impl.Layout wraps ov::Layout";

    layout.def(py::init<>());
    layout.def(py::init<const char*>());
    layout.def(py::init<const std::string&>());

    // operator overloading
    layout.def(py::self == py::self);
    layout.def(py::self != py::self);

    layout.def("scalar", &ov::Layout::scalar);
    layout.def("has_name", &ov::Layout::has_name);
    layout.def("get_index_by_name", &ov::Layout::get_index_by_name);
    layout.def("to_string", &ov::Layout::to_string);
}
