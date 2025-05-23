# OpenVINO Release Management
The process described below reflects the approach to managing OpenVINO releases.

## Release Milestones
- Planning
- Execution (development of new features)
- Stabilization (Feature Freeze, Code Freeze milestones)
- Validation
- Distribution

### Planning
This phase takes 2-4 weeks and involves scoping the backlog, prioritizing it, analyzing, and making commitments by developers for timelines specified by the release manager.

### Execution (development of new features)
- [OpenVINO Contributing Guide](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md)
- [Code Contribution Guide](https://docs.openvino.ai/2025/about-openvino/contributing/code-contribution-guide.html)
- [OpenVINO First Good Issue](https://github.com/openvinotoolkit/openvino/issues/17502)

### Stabilization (Feature Freeze, Code Freeze milestones)
- **Feature Freeze**: This milestone ensures that no new features are added to the software after a certain point. This allows the development team to focus on stabilizing and refining the existing features, fixing bugs, and improving performance without the risk of introducing new issues.
- **Code Freeze**: This milestone marks the point where no new code changes are allowed except for critical bug fixes. This helps in ensuring that the final product is stable and reliable, as it minimizes the risk of last-minute changes that could introduce new bugs or instability.

### Release Validation
- This is a continuous process executed on a regular basis with cadence based on testing type: nightly, bi-weekly, weekly.
- After Code Freeze, the testing team can perform final regression testing to ensure that recent changes have not introduced new bugs and that the software meets the required quality standards.

### Distribution
- OpenVINO has different types of build distribution: Regular releases, Long-Term Support, Pre-release releases, Nightly builds. Read more here: [OpenVINO Release Policy](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/release-policy.html)
- Different distribution channels are supported. Explore different options here: [OpenVINO Download](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)
