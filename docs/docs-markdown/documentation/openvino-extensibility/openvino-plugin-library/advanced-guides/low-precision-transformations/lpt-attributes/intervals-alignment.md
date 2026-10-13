---
sidebar_position: 2
sidebar_label: "IntervalsAlignment Attribute"
description: "Learn about IntervalsAlignment attribute, which describes a subgraph with the same quantization intervals alignment."
---

# IntervalsAlignment Attribute

`ov::IntervalsAlignmentAttribute` class represents the `IntervalsAlignment` attribute.

The attribute defines a subgraph with the same quantization intervals alignment. `FakeQuantize` operations are included. The attribute is used by quantization operations.

| Property name | Values |
| --- | --- |
| Required | Yes |
| Defined | Operation |
| Properties | combined interval, minimal interval, minimal levels, preferable precisions |
