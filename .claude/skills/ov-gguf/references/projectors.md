# Projector support

Read the relevant support and graph-contract sections of
[mmproj.md](../../../../src/frontends/gguf/docs/mmproj.md). `clip` selects the mmproj coordinator;
support for a language backbone does not establish support for its media encoders or preprocessing.

Use `ProjectorDefinition` through `ProjectorExtension` to add one vision/audio branch.
It extends the separate modality/projector registry. Read
[component extensions](../../../../src/frontends/gguf/docs/extensions.md#extend-mmproj-with-a-projector-component)
for selection, replacement, and packaging.

- Reuse `build_builtin_projector()` only when metadata, weights, and computation fit that topology.
- For a new topology, build the branch in the coordinator's `GgufGraphContext` and return
  `ProjectorResult`. Leave primary output naming and `finish()` to the coordinator.
- Use explicit component replacement when overriding an existing projector. Another whole-model
  `clip` handler can overlap with the built-in coordinator. A genuinely different whole-model
  format can use [architecture support](architectures.md).
- Register before `load()` on the frontend that converts the file; loaded input models retain
  their registry snapshot. Combined files must preserve every declared modality.

Validate raw and adapted embeddings against the llama.cpp CPU encoder using
[mmproj fixtures](../../../../src/frontends/gguf/tests/test_data/mmproj_accuracy/README.md).
For dynamic encoders, exercise multiple sizes on one compiled model. Cover combined modalities
when applicable, and compare real checkpoints with matching preprocessing and positional inputs.
Run adaptation on independent model clones when extracting both vision and audio.
