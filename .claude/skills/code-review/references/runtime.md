# Runtime and Core Review

For graph passes and the pass/pattern infrastructure, also apply the
[transformation review guidance](transformations.md).

- For stateful helpers or cached parameters, verify initialization, update,
  and reset behavior across all inference paths.
