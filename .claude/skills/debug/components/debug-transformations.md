The debug and troubleshooting of OpenVINO transformations can be performed using various debug capabilities activated via environment variables.

# Prerequisites
Matcher logging requires `-DENABLE_OPENVINO_DEBUG=ON`. Verify in CMakeCache.txt before suggesting `OV_MATCHER_LOGGING`.

# Reference
Read `src/common/transformations/docs/debug_capabilities/README.md` — use the "When to use" guidance to match the observed problem to the right capability.

# Handoff
If the symptom is specifically a MatcherPass not firing (transformation not applied, pattern not matching, callback never called), hand off to the `/debug-matcher-pass` skill — it provides a deeper automated diagnosis workflow with matcher log analysis and reproducer test generation.

# Steps
1. Read the debug capabilities README
2. Match the observed symptom to the relevant "When to use" entries
3. If the issue is a MatcherPass not firing — invoke `/debug-matcher-pass` instead of continuing
4. Read the linked detail doc for the chosen capability
5. Use suitable environment variables
