# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from readd_transient_failure import (
    BOT_LOGIN,
    QUEUE_EVENTS_QUERY,
    dequeue_reason,
    find_eligibility_marker,
    latest_queue_events,
    marker_is_current,
    marker_matches_analysis_run,
    parse_marker,
    pr_number_from_event,
    process_event,
    source_run_matches_pr,
)


HEAD_SHA = "a" * 40
MARKER_BODY = (
    "<!-- ci-doctor-mq-eligible pr=123 head_sha="
    f"{HEAD_SHA} source_run_id=456 analysis_run_id=789 -->"
)
NOW = datetime(2026, 11, 6, 8, 0, tzinfo=timezone.utc)
OTHER_REMOVAL_REASONS = (
    "UNKNOWN_REMOVAL_REASON",
    "MANUAL",
    "MERGE",
    "MERGE_CONFLICT",
    "CI_TIMEOUT",
    "ALREADY_MERGED",
    "QUEUE_CLEARED",
    "ROLL_BACK",
    "BRANCH_PROTECTIONS",
    "GIT_TREE_INVALID",
    "INVALID_MERGE_COMMIT",
    "failed_checks",
    "ci_failure",
    "UNRECOGNIZED_REASON",
)


class MarkerTests(unittest.TestCase):
    def test_parse_valid_marker(self):
        marker = parse_marker(MARKER_BODY, NOW)

        self.assertEqual(marker.pr_number, "123")
        self.assertEqual(marker.head_sha, HEAD_SHA)
        self.assertEqual(marker.source_run_id, "456")
        self.assertEqual(marker.analysis_run_id, "789")

    def test_parse_rejects_malformed_marker(self):
        self.assertIsNone(parse_marker("<!-- ci-doctor-mq-eligible pr=123 -->", NOW))

    def test_only_trusted_author_and_current_head_are_accepted(self):
        comments = [
            SimpleNamespace(
                user=SimpleNamespace(login="contributor"),
                body=MARKER_BODY,
                created_at=NOW,
            ),
            SimpleNamespace(
                user=SimpleNamespace(login=BOT_LOGIN),
                body=MARKER_BODY.replace(HEAD_SHA, "b" * 40),
                created_at=NOW,
            ),
            SimpleNamespace(
                user=SimpleNamespace(login=BOT_LOGIN),
                body=MARKER_BODY,
                created_at=NOW,
            ),
        ]

        marker = find_eligibility_marker(comments, 123, HEAD_SHA)

        self.assertIsNotNone(marker)
        self.assertEqual(marker.source_run_id, "456")

    def test_user_authored_marker_is_rejected(self):
        comment = SimpleNamespace(
            user=SimpleNamespace(login="contributor"),
            body=MARKER_BODY,
            created_at=NOW,
        )

        self.assertIsNone(find_eligibility_marker([comment], 123, HEAD_SHA))

    def test_stale_marker_before_latest_enqueue_is_rejected(self):
        marker = parse_marker(MARKER_BODY, NOW)

        self.assertFalse(marker_is_current(marker, NOW))
        self.assertTrue(marker_is_current(marker, NOW.replace(hour=7)))
        self.assertFalse(marker_is_current(marker, None))

    def test_completion_fallback_must_match_marker_analysis_run(self):
        marker = parse_marker(MARKER_BODY, NOW)

        self.assertTrue(marker_matches_analysis_run(marker, {"id": 789}))
        self.assertFalse(marker_matches_analysis_run(marker, {"id": 790}))


class EventTests(unittest.TestCase):
    def test_dequeued_event_reads_top_level_reason(self):
        self.assertEqual(dequeue_reason({"reason": "CI_FAILURE"}), "CI_FAILURE")
        self.assertEqual(dequeue_reason({"reason": "MANUAL"}), "MANUAL")

    def test_dequeued_event_requires_non_empty_string_reason(self):
        invalid_events = [
            {},
            {"pull_request": {"reason": "CI_FAILURE"}},
            {"reason": None},
            {"reason": ""},
            {"reason": 123},
            {"reason": ["CI_FAILURE"]},
        ]
        for event in invalid_events:
            with self.subTest(event=event):
                with self.assertRaisesRegex(SystemExit, "top-level reason string"):
                    dequeue_reason(event)

    def test_other_removal_reasons_are_ignored_before_pr_lookup(self):
        for reason in OTHER_REMOVAL_REASONS:
            with self.subTest(reason=reason):
                repo = Mock()
                process_event({"action": "dequeued", "reason": reason}, Mock(), repo, "owner/repo")
                repo.get_pull.assert_not_called()

    def test_dequeue_without_required_reason_fails_before_pr_lookup(self):
        repo = Mock()
        with self.assertRaisesRegex(SystemExit, "top-level reason string"):
            process_event({"action": "dequeued"}, Mock(), repo, "owner/repo")
        repo.get_pull.assert_not_called()

    def test_ci_failure_dequeue_checks_marker(self):
        pull = Mock()
        pull.number = 123
        pull.head.sha = HEAD_SHA
        pull.get_issue_comments.return_value = []
        repo = Mock()
        repo.get_pull.return_value = pull

        process_event(
            {"action": "dequeued", "reason": "CI_FAILURE", "pull_request": {"number": 123}},
            Mock(),
            repo,
            "owner/repo",
        )

        repo.get_pull.assert_called_once_with(123)

    def test_analysis_completion_does_not_require_webhook_reason(self):
        pull = Mock()
        pull.number = 123
        pull.head.sha = HEAD_SHA
        pull.get_issue_comments.return_value = []
        repo = Mock()
        repo.get_pull.return_value = pull

        process_event(
            {
                "action": "completed",
                "workflow_run": {
                    "name": "CI Failure Doctor — Merge Queue",
                    "event": "workflow_run",
                    "conclusion": "success",
                    "display_title": "CI Doctor MQ: gh-readonly-queue/master/pr-123-abcdef123456",
                },
            },
            Mock(),
            repo,
            "owner/repo",
        )

        repo.get_pull.assert_called_once_with(123)

    def test_dequeued_event_resolves_pr_number(self):
        self.assertEqual(
            pr_number_from_event({"action": "dequeued", "pull_request": {"number": 123}}),
            "123",
        )

    def test_analysis_completion_resolves_pr_number_from_run_title(self):
        event = {
            "workflow_run": {
                "name": "CI Failure Doctor — Merge Queue",
                "event": "workflow_run",
                "conclusion": "success",
                "display_title": "CI Doctor MQ: gh-readonly-queue/master/pr-123-abcdef123456",
            }
        }

        self.assertEqual(pr_number_from_event(event), "123")

    def test_analysis_completion_rejects_non_analysis_workflow(self):
        with self.assertRaisesRegex(SystemExit, "did not originate"):
            pr_number_from_event({"workflow_run": {"name": "Other", "conclusion": "success"}})

    def test_source_run_must_be_failed_merge_group_for_same_pr(self):
        valid = SimpleNamespace(
            event="merge_group",
            conclusion="failure",
            head_branch="gh-readonly-queue/master/pr-123-abcdef123456",
        )
        wrong_pr = SimpleNamespace(
            event="merge_group",
            conclusion="failure",
            head_branch="gh-readonly-queue/master/pr-124-abcdef123456",
        )

        self.assertTrue(source_run_matches_pr(valid, "123"))
        self.assertFalse(source_run_matches_pr(wrong_pr, "123"))
        self.assertFalse(source_run_matches_pr(SimpleNamespace(
            event="pull_request",
            conclusion="failure",
            head_branch=valid.head_branch,
        ), "123"))

    def test_latest_queue_event_includes_removal_reason_and_enqueue_time(self):
        added_at = NOW.replace(hour=6)
        removed_at = NOW.replace(hour=7)
        gh = Mock()
        gh.requester.requestJsonAndCheck.return_value = (None, {
            "data": {
                "repository": {
                    "pullRequest": {
                        "timelineItems": {
                            "nodes": [
                                {"__typename": "AddedToMergeQueueEvent", "createdAt": added_at.isoformat()},
                                {
                                    "__typename": "RemovedFromMergeQueueEvent",
                                    "createdAt": removed_at.isoformat(),
                                    "reason": "CI_FAILURE",
                                },
                            ],
                            "pageInfo": {"hasPreviousPage": False, "startCursor": None},
                        }
                    }
                }
            }
        })

        latest, latest_added = latest_queue_events(gh, "owner/repo", 123)

        self.assertEqual(latest, ("removed_from_merge_queue", removed_at, "CI_FAILURE"))
        self.assertEqual(latest_added, added_at)
        request = gh.requester.requestJsonAndCheck.call_args
        self.assertEqual(request.args[:2], ("POST", "/graphql"))
        self.assertIn("RemovedFromMergeQueueEvent", request.kwargs["input"]["query"])
        self.assertEqual(request.kwargs["input"]["variables"]["number"], 123)

    def test_latest_queue_event_paginates_until_latest_admission(self):
        added_at = NOW.replace(hour=6)
        removed_at = NOW.replace(hour=7)
        gh = Mock()
        gh.requester.requestJsonAndCheck.side_effect = [
            (None, {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "timelineItems": {
                                "nodes": [{
                                    "__typename": "RemovedFromMergeQueueEvent",
                                    "createdAt": removed_at.isoformat(),
                                    "reason": "CI_FAILURE",
                                }],
                                "pageInfo": {"hasPreviousPage": True, "startCursor": "cursor-2"},
                            }
                        }
                    }
                }
            }),
            (None, {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "timelineItems": {
                                "nodes": [{
                                    "__typename": "AddedToMergeQueueEvent",
                                    "createdAt": added_at.isoformat(),
                                }],
                                "pageInfo": {"hasPreviousPage": False, "startCursor": "cursor-1"},
                            }
                        }
                    }
                }
            }),
        ]

        latest, latest_added = latest_queue_events(gh, "owner/repo", 123)

        self.assertEqual(latest, ("removed_from_merge_queue", removed_at, "CI_FAILURE"))
        self.assertEqual(latest_added, added_at)
        second_variables = gh.requester.requestJsonAndCheck.call_args_list[1].kwargs["input"]["variables"]
        self.assertEqual(second_variables["before"], "cursor-2")

    def test_graphql_queue_reason_can_be_null(self):
        gh = Mock()
        gh.requester.requestJsonAndCheck.return_value = (None, {
            "data": {
                "repository": {
                    "pullRequest": {
                        "timelineItems": {
                            "nodes": [{
                                "__typename": "RemovedFromMergeQueueEvent",
                                "createdAt": NOW.isoformat(),
                                "reason": None,
                            }],
                            "pageInfo": {"hasPreviousPage": False, "startCursor": None},
                        }
                    }
                }
            }
        })

        latest, latest_added = latest_queue_events(gh, "owner/repo", 123)

        self.assertEqual(latest, ("removed_from_merge_queue", NOW, ""))
        self.assertIsNone(latest_added)


class ReaddDecisionTests(unittest.TestCase):
    def setUp(self):
        self.pull = Mock()
        self.pull.number = 123
        self.pull.head.sha = HEAD_SHA
        self.pull.get_issue_comments.return_value = [
            SimpleNamespace(user=SimpleNamespace(login=BOT_LOGIN), body=MARKER_BODY, created_at=NOW)
        ]
        self.removal = SimpleNamespace(
            event="removed_from_merge_queue",
            reason="CI_FAILURE",
            created_at=NOW.replace(hour=7),
        )
        self.pull.as_issue.return_value.get_timeline.return_value = [
            SimpleNamespace(event="added_to_merge_queue", created_at=NOW.replace(hour=6)),
            self.removal,
        ]
        self.repo = Mock()
        self.repo.get_pull.return_value = self.pull
        self.repo.get_workflow_run.return_value = SimpleNamespace(
            event="merge_group",
            conclusion="failure",
            head_branch="gh-readonly-queue/master/pr-123-abcdef123456",
        )
        self.event = {"action": "dequeued", "reason": "CI_FAILURE", "pull_request": {"number": 123}}
        self.gh = Mock()
        self.readd = self._patch("readd_to_queue")
        self.timeline = self._patch("latest_queue_events")
        self.timeline.return_value = (
            ("removed_from_merge_queue", NOW.replace(hour=7), "CI_FAILURE"),
            NOW.replace(hour=6),
        )
        self._patch("merge_queue_status").return_value = "not_in_queue"
        self._patch("require_env").return_value = "test-token"

    def _patch(self, name):
        patcher = patch(f"readd_transient_failure.{name}")
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def test_ci_failure_with_valid_marker_reaches_enqueue(self):
        process_event(self.event, self.gh, self.repo, "owner/repo")

        self.readd.assert_called_once_with(
            self.pull, "123", "owner/repo", "transient merge-queue failure", "test-token", self.gh, self.repo
        )

    def test_dequeued_webhook_reason_is_authoritative_when_graphql_reason_is_null(self):
        self.timeline.return_value = (
            ("removed_from_merge_queue", NOW.replace(hour=7), ""),
            NOW.replace(hour=6),
        )

        process_event(self.event, self.gh, self.repo, "owner/repo")

        self.readd.assert_called_once()

    def test_dequeued_webhook_rejects_conflicting_graphql_reason(self):
        for reason in OTHER_REMOVAL_REASONS:
            with self.subTest(reason=reason):
                self.timeline.return_value = (
                    ("removed_from_merge_queue", NOW.replace(hour=7), reason),
                    NOW.replace(hour=6),
                )
                process_event(self.event, self.gh, self.repo, "owner/repo")
                self.readd.assert_not_called()

    def test_ci_failure_still_requires_a_trusted_marker(self):
        self.pull.get_issue_comments.return_value = []

        process_event(self.event, self.gh, self.repo, "owner/repo")

        self.readd.assert_not_called()

    def test_analysis_completion_checks_ci_failure_without_webhook_reason(self):
        event = {
            "action": "completed",
            "workflow_run": {
                "id": 789,
                "name": "CI Failure Doctor — Merge Queue",
                "event": "workflow_run",
                "conclusion": "success",
                "display_title": "CI Doctor MQ: gh-readonly-queue/master/pr-123-abcdef123456",
            },
        }

        process_event(event, self.gh, self.repo, "owner/repo")

        self.readd.assert_called_once()
        self.readd.reset_mock()
        self.timeline.return_value = (
            ("removed_from_merge_queue", NOW.replace(hour=7), "MANUAL"),
            NOW.replace(hour=6),
        )
        process_event(event, self.gh, self.repo, "owner/repo")
        self.readd.assert_not_called()

    def test_analysis_completion_fails_closed_when_graphql_reason_is_missing(self):
        event = {
            "action": "completed",
            "workflow_run": {
                "id": 789,
                "name": "CI Failure Doctor — Merge Queue",
                "event": "workflow_run",
                "conclusion": "success",
                "display_title": "CI Doctor MQ: gh-readonly-queue/master/pr-123-abcdef123456",
            },
        }
        self.timeline.return_value = (
            ("removed_from_merge_queue", NOW.replace(hour=7), ""),
            NOW.replace(hour=6),
        )

        process_event(event, self.gh, self.repo, "owner/repo")

        self.readd.assert_not_called()


if __name__ == "__main__":
    unittest.main()
