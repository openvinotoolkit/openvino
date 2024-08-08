// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/experimental/experimental.hpp"

#include "openvino/core/bound_evaluation_util.hpp"
#include "openvino/core/validation_util.hpp"

namespace py = pybind11;

void regmodule_experimental(py::module m) {
    py::module m_dev = m.def_submodule("experimental", "openvino.experimental submodule");

    m_dev.def("evaluate_as_partial_shape",
              &ov::util::evaluate_as_partial_shape,
              py::arg("output"),
              py::arg("partial_shape"),
              R"(
                    Evaluates lower and upper value estimations for the output tensor. 
                    The estimation will be represented as a partial shape object, 
                    using Dimension(min, max) for each element.

                    :param output: Node output pointing to the tensor for estimation.
                    :type output: openvino.runtime.Output
                    :param partial_shape: The resulting estimation will be stored in this PartialShape.
                    :type partial_shape: openvino.PartialShape
                    :return: True if estimation evaluation was successful, false otherwise.
                    :rtype: bool
                )");
    m_dev.def("evaluate_both_bounds",
              &ov::util::evaluate_both_bounds,
              py::arg("output"),
              R"(
                    Evaluates lower and upper value estimations of the output tensor.
                    It traverses the graph upwards to deduce the estimation.

                    :param output: Node output pointing to the tensor for estimation.
                    :type output: openvino.runtime.Output
                    :return: Tensors representing the lower and upper bound value estimations.
                    :rtype: Tuple[openvino.Tensor, openvino.Tensor]
                )");
}
