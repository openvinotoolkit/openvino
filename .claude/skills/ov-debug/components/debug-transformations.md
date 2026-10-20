The debug and troubleshooting of OpenVINO transformations can be performed using various debug capabilities activated via environment variables.

# Reference
Read [src/common/transformations/docs/debug_capabilities/README.md](../../../../src/common/transformations/docs/debug_capabilities/README.md) — use the "When to use" guidance to match the observed problem to the right capability.

# Specialized diagnosis
For a MatcherPass that does not fire (transformation not applied, pattern not matching, callback never called), consider the [`ov-debug-matcher-pass`](../../ov-debug-matcher-pass/SKILL.md) skill for deeper diagnosis with matcher log analysis and reproducer test generation. The debug capabilities below can also help investigate these symptoms.

# Steps
1. Read the debug capabilities README
2. Match the observed symptom to the relevant "When to use" entries
3. For a MatcherPass that does not fire, consider the specialized diagnosis skill above
4. Read the linked detail doc for the chosen capability
5. Use suitable environment variables
