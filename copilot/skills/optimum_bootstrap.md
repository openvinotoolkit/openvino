# Skill: Optimum-Intel Bootstrap

**Trigger:** Called automatically before every optimum-intel task.

## Purpose

Ensure the agent has a local, up-to-date optimum-intel clone and has loaded the
upstream SKILL.md reference document that defines conventions, patterns and
code-level details.

## Steps

1. **Clone or locate optimum-intel:**
   ```bash
   # If not already present in the working directory
   git clone https://github.com/huggingface/optimum-intel.git /tmp/optimum-intel
   cd /tmp/optimum-intel
   ```

2. **Load the SKILL.md reference document:**
   Try `main` branch first:
   ```bash
   git checkout main && git pull
   ```
   Look for the skill file at:
   - `skills/adding-new-model-support/SKILL.md`
   - `skills/SKILL.md`

   If the file does not exist on `main`, fetch PR #1616:
   ```bash
   git fetch origin pull/1616/head:pr-1616
   git checkout pr-1616
   ```
   Then read `skills/adding-new-model-support/SKILL.md`.

3. **Read the SKILL.md file fully.** It contains:
   - The complete workflow for adding new model support
   - Model architecture analysis patterns
   - Model config class conventions (`model_configs.py`)
   - Model patching patterns (`model_patcher.py`) including MoE vectorization
   - Test file locations and conventions
   - Documentation update procedures
   - Reference PRs to study (e.g., #1569 for Afmoe)

4. **Keep the clone available** - it is needed throughout the task for reading
   source files, studying patterns, running exports, and creating patches.

## External References

- **optimum-intel repository:** https://github.com/huggingface/optimum-intel
- **SKILL.md PR:** https://github.com/huggingface/optimum-intel/pull/1616
- **Reference PR (Afmoe):** https://github.com/huggingface/optimum-intel/pull/1569
