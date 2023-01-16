# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helper classes for aggregating TensorIterator input/output desciptor attributes."""

from typing import List

from ngraph.impl import Node
from ngraph.impl.op import Parameter


class GraphBody(object):
    """Class containing graph parameters and results."""

    def __init__(
        self,
        parameters: List[Parameter],
        results: List[Node],
    ) -> None:
        self.parameters = parameters
        self.results = results

    def serialize(self) -> dict:
        """Serialize GraphBody as a dictionary."""
        return {
            "parameters": self.parameters,
            "results": self.results,
        }


class TensorIteratorInputDesc(object):
    """Represents a generic input descriptor for TensorIterator operator."""

    def __init__(
        self,
        input_idx: int,
        body_parameter_idx: int,
    ) -> None:
        self.input_idx = input_idx
        self.body_parameter_idx = body_parameter_idx

    def serialize(self) -> dict:
        """Serialize TensorIteratorInputDesc as a dictionary."""
        return {
            "input_idx": self.input_idx,
            "body_parameter_idx": self.body_parameter_idx,
        }


class TensorIteratorSliceInputDesc(TensorIteratorInputDesc):
    """Represents a TensorIterator graph body input formed from slices of TensorIterator input."""

    def __init__(
        self,
        input_idx: int,
        body_parameter_idx: int,
        start: int,
        stride: int,
        part_size: int,
        end: int,
        axis: int,
    ) -> None:
        super().__init__(input_idx, body_parameter_idx)
        self.start = start
        self.stride = stride
        self.part_size = part_size
        self.end = end
        self.axis = axis

    def serialize(self) -> dict:
        """Serialize TensorIteratorSliceInputDesc as a dictionary."""
        output = super().serialize()
        output["start"] = self.start
        output["stride"] = self.stride
        output["part_size"] = self.part_size
        output["end"] = self.end
        output["axis"] = self.axis
        return output


class TensorIteratorMergedInputDesc(TensorIteratorInputDesc):
    """Represents a TensorIterator graph body input with initial value in the first iteration.

    Later on, this input value is computed inside graph body.
    """

    def __init__(
        self,
        input_idx: int,
        body_parameter_idx: int,
        body_value_idx: int,
    ) -> None:
        super().__init__(input_idx, body_parameter_idx)
        self.body_value_idx = body_value_idx

    def serialize(self) -> dict:
        """Serialize TensorIteratorMergedInputDesc as a dictionary."""
        output = super().serialize()
        output["body_value_idx"] = self.body_value_idx
        return output


class TensorIteratorInvariantInputDesc(TensorIteratorInputDesc):
    """Represents a TensorIterator graph body input that has invariant value during iteration."""

    def __init__(
        self,
        input_idx: int,
        body_parameter_idx: int,
    ) -> None:
        super().__init__(input_idx, body_parameter_idx)


class TensorIteratorOutputDesc(object):
    """Represents a generic output descriptor for TensorIterator operator."""

    def __init__(
        self,
        body_value_idx: int,
        output_idx: int,
    ) -> None:
        self.body_value_idx = body_value_idx
        self.output_idx = output_idx

    def serialize(self) -> dict:
        """Serialize TensorIteratorOutputDesc as a dictionary."""
        return {
            "body_value_idx": self.body_value_idx,
            "output_idx": self.output_idx,
        }


class TensorIteratorBodyOutputDesc(TensorIteratorOutputDesc):
    """Represents an output from a specific iteration."""

    def __init__(
        self,
        body_value_idx: int,
        output_idx: int,
        iteration: int = -1,
    ) -> None:
        super().__init__(body_value_idx, output_idx)
        self.iteration = iteration

    def serialize(self) -> dict:
        """Serialize TensorIteratorBodyOutputDesc as a dictionary."""
        output = super().serialize()
        output["iteration"] = self.iteration
        return output


class TensorIteratorConcatOutputDesc(TensorIteratorOutputDesc):
    """Represents an output produced by concatenation of output from each iteration."""

    def __init__(
        self,
        body_value_idx: int,
        output_idx: int,
        start: int,
        stride: int,
        part_size: int,
        end: int,
        axis: int,
    ) -> None:
        super().__init__(body_value_idx, output_idx)
        self.start = start
        self.stride = stride
        self.part_size = part_size
        self.end = end
        self.axis = axis

    def serialize(self) -> dict:
        """Serialize TensorIteratorConcatOutputDesc as a dictionary."""
        output = super().serialize()
        output["start"] = self.start
        output["stride"] = self.stride
        output["part_size"] = self.part_size
        output["end"] = self.end
        output["axis"] = self.axis
        return output
