# Broadcast rules for elementwise ops {#openvino_docs_ops_broadcast_rules}

The purpose of this document is to provide set of common rules which are applicable for ops using broadcasting.

## Rules

1. * **Broadcast range of values**:
    * *none* - no auto-broadcasting is allowed, all input shapes must match,
    * *numpy* - numpy broadcasting rules, description is available in [Broadcast_1](movement/Broadcast_1.md).
