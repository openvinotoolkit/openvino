---
description: |
  Gate automatic CI Doctors before activation: investigate only when the target
  run's Overall_Status job (displayed as ci/gha_overall_status*) failed.
jobs:
  check_overall_status:
    name: Check required pipeline status
    if: ${{ github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      actions: read
      contents: read
    outputs:
      should_investigate: ${{ steps.overall_status.outputs.should_investigate }}
    steps:
      - name: Checkout CI Doctor gate
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          persist-credentials: false
          sparse-checkout: .github/scripts/agentic-workflows
      - name: Check Overall_Status
        id: overall_status
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          REPO: ${{ github.repository }}
          RUN_ID: ${{ github.event.workflow_run.id || github.event.inputs.run_id }}
          RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}
          LINK: ${{ github.event.inputs.link }}
        run: python3 .github/scripts/agentic-workflows/check_overall_status.py
---
