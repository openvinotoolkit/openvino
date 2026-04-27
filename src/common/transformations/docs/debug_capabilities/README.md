# Debug capabilities
Debug capabilities are the set of useful debug features for OpenVINO transformations, controlled by environment variables.

They can be activated at runtime and are useful for analyzing transformation behavior, profiling pass execution, and inspecting model state between passes.

* [Matcher logging](matcher_logging.md)
  When to use: a MatcherPass transformation is not firing or matching unexpectedly — logs the pattern matching process to show why matches succeed or fail.
  Requires: `-DENABLE_OPENVINO_DEBUG=ON`
  Example: `OV_MATCHER_LOGGING=true OV_MATCHERS_TO_LOG=EliminateSplitConcat ./your_program`

* [Transformation profiling](transformation_profiling.md)
  When to use: slow model compilation or need to identify which transformation passes take the most time.
  Example: `OV_ENABLE_PROFILE_PASS=true`

* [Model visualization](model_visualization.md)
  When to use: need to see the model graph structure after specific passes — generates .svg files.
  Example: `OV_ENABLE_VISUALIZE_TRACING=true` or `OV_ENABLE_VISUALIZE_TRACING="Pass1,Pass2"`

* [Model serialization](model_serialization.md)
  When to use: need to inspect model IR (.xml/.bin) after specific passes — useful for diffing model state before and after a transformation.
  Example: `OV_ENABLE_SERIALIZE_TRACING=true` or `OV_ENABLE_SERIALIZE_TRACING="Pass1,Pass2"`

## See also

* [debug-matcher-pass skill](../../../../.claude/skills/debug-matcher-pass/SKILL.md) — automated diagnosis workflow for MatcherPass transformations that are not firing. Collects matcher logs, identifies root cause, and generates a reproducer test.
