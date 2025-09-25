const core = require('@actions/core');
const github = require('@actions/github');

/**
 * Wait for a specific check to complete
 * @param {Object} octokit - GitHub API client
 * @param {string} owner - Repository owner
 * @param {string} repo - Repository name
 * @param {string} ref - Git reference (commit SHA)
 * @param {string} checkName - Name of the check to wait for
 * @param {number} waitInterval - Wait interval in seconds
 * @param {number} timeout - Maximum timeout in seconds
 * @returns {Promise<Object>} Check result
 */
async function waitForCheck(octokit, owner, repo, ref, checkName, waitInterval, timeout) {
    const startTime = Date.now();
    const timeoutMs = timeout * 1000;
    const waitIntervalMs = waitInterval * 1000;

    core.info(`Waiting for check "${checkName}" on ref "${ref}" to complete...`);
    core.info(`Timeout: ${timeout}s, Check interval: ${waitInterval}s`);

    while (Date.now() - startTime < timeoutMs) {
        try {
            // Get check runs for the specific commit
            const { data: checkRuns } = await octokit.rest.checks.listForRef({
                owner,
                repo,
                ref,
                check_name: checkName,
                per_page: 100
            });

            core.debug(`Found ${checkRuns.check_runs.length} check runs for "${checkName}"`);

            if (checkRuns.check_runs.length === 0) {
                core.info(`No check runs found for "${checkName}" yet, waiting...`);
            } else {
                // Find the most recent check run
                const latestCheckRun = checkRuns.check_runs.sort((a, b) =>
                    new Date(b.started_at) - new Date(a.started_at)
                )[0];

                core.info(`Check "${checkName}" status: ${latestCheckRun.status}, conclusion: ${latestCheckRun.conclusion || 'N/A'}`);

                if (latestCheckRun.status === 'completed') {
                    core.info(`Check "${checkName}" completed with conclusion: ${latestCheckRun.conclusion}`);
                    return {
                        status: latestCheckRun.status,
                        conclusion: latestCheckRun.conclusion,
                        checkRun: latestCheckRun
                    };
                }

                if (latestCheckRun.status === 'in_progress') {
                    core.info(`Check "${checkName}" is still in progress...`);
                } else if (latestCheckRun.status === 'queued') {
                    core.info(`Check "${checkName}" is queued...`);
                }
            }
        } catch (error) {
            core.warning(`Error fetching check runs: ${error.message}`);

            // If it's a 404, the check might not exist yet
            if (error.status === 404) {
                core.info('Check not found yet, continuing to wait...');
            } else {
                // For other errors, we might want to retry but log the error
                core.error(`API error: ${error.message}`);
            }
        }

        core.info(`Waiting ${waitInterval} seconds before next check...`);
        await new Promise(resolve => setTimeout(resolve, waitIntervalMs));
    }

    throw new Error(`Timeout: Check "${checkName}" did not complete within ${timeout} seconds`);
}

/**
 * Main function
 */
async function run() {
    try {
        // Get inputs
        const ref = core.getInput('ref', { required: true });
        const checkName = core.getInput('check-name', { required: true });
        const token = core.getInput('repo-token', { required: true });
        const waitInterval = parseInt(core.getInput('wait-interval') || '10', 10);
        const timeout = parseInt(core.getInput('timeout') || '600', 10);

        // Get repository info (default to current repository)
        const owner = core.getInput('owner') || github.context.repo.owner;
        const repo = core.getInput('repo') || github.context.repo.repo;

        core.info(`Repository: ${owner}/${repo}`);
        core.info(`Reference: ${ref}`);
        core.info(`Check name: ${checkName}`);

        // Validate inputs
        if (waitInterval < 1 || waitInterval > 300) {
            throw new Error('wait-interval must be between 1 and 300 seconds');
        }

        if (timeout < 1 || timeout > 3600) {
            throw new Error('timeout must be between 1 and 3600 seconds');
        }

        // Create GitHub API client
        const octokit = github.getOctokit(token);

        // Wait for the check to complete
        const result = await waitForCheck(octokit, owner, repo, ref, checkName, waitInterval, timeout);

        // Set outputs
        core.setOutput('status', result.status);
        core.setOutput('conclusion', result.conclusion);

        core.info(`Check "${checkName}" completed successfully`);
        core.info(`Status: ${result.status}`);
        core.info(`Conclusion: ${result.conclusion}`);

        // Exit with appropriate code based on conclusion
        if (result.conclusion === 'success') {
            core.info('Check completed with successful conclusion');
        } else if (result.conclusion === 'failure' || result.conclusion === 'cancelled' || result.conclusion === 'timed_out' || result.conclusion === 'neutral' || result.conclusion === 'skipped') {
            core.setFailed(`Check completed with conclusion: ${result.conclusion}`);
        } else if (result.conclusion === 'action_required') {
            core.setFailed(`Check requires action: ${result.conclusion}`);
        } else {
            core.warning(`Check completed with unknown conclusion: ${result.conclusion}`);
        }
    } catch (error) {
        core.setFailed(error.message);
        core.error(error.stack || error.toString());
    }
}

// Only run if this file is executed directly (not imported)
if (require.main === module) {
    run();
}

module.exports = { run, waitForCheck };