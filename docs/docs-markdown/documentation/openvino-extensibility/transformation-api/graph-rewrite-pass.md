---
sidebar_position: 3
sidebar_label: "OpenVINO Graph Rewrite Pass"
description: "Get to know how Graph Rewrite handles running multiple matcher passes on ov::Model in a single graph traversal."
---

# OpenVINO Graph Rewrite Pass

`ov::pass::GraphRewrite` serves for running multiple matcher passes on `ov::Model` in a single graph traversal.
Example:

```cpp
    // Two matcher passes will run simultaneously in a single graph traversal
    ov::pass::GraphRewrite pass;
    pass.add_matcher<ov::pass::DecomposeDivideMatcher>();
    pass.add_matcher<ov::pass::ReluReluFusionMatcher>();
    pass.run_on_model(f);
```

In addition, GraphRewrite handles nodes that were registered by MatcherPasses during their execution. This nodes will be added to the beginning of the sequence with nodes for pattern matching.

:::note
When using `ov::pass::Manager` temporary GraphRewrite is used to execute single MatcherPass.
:::

GraphRewrite has two algorithms for MatcherPasses execution. First algorithm is straightforward. It applies each MatcherPass in registration order to current node.

![](graph_rewrite_execution.png)

But it is not really efficient when you have a lot of registered passes. So first of all GraphRewrite checks that all MatcherPass patterns has type-based root node (it means that type of this node is not hidden into predicate).
And then creates map from registered MatcherPasses. That helps to avoid additional cost of applying each MatcherPass for each node.

![](graph_rewrite_efficient_search.png)

:::note
GraphRewrite execution algorithm cannot be set manually and depends only on root nodes registered inside MatcherPasses.
:::

## See Also

- [OpenVINO™ Transformations](index.md)
