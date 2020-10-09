#!/usr/bin/python3

# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import requests


class MSTeamsCommunicator:
    """Class communicating with MSTeams using Incoming Webhook.

    The purpose of this class is to use MSTeams API to send message.
    Docs for used API, including wrapped methods can be found at:
    https://docs.microsoft.com/en-us/outlook/actionable-messages/send-via-connectors
    """

    def __init__(self, _ci_alerts_channel_url):
        self._ci_alerts_channel_url = _ci_alerts_channel_url
        self._queued_messages = {
            self._ci_alerts_channel_url: [],
        }

    @property
    def messages(self):
        """
        Get list of queued messages.

            :return:           List of queued messages
            :return type:      List[String]
        """
        return self._queued_messages.values()

    def queue_message(self, message):
        """
        Queue message to be sent later.

            :param message:     Message content
            :type message:      String
        """
        self._queued_messages[self._ci_alerts_channel_url].append(message)

    def _parse_text(self, watchdog_log, message):
        """
        Parse text to display as alert.

            :param watchdog_log:     Watchdog log content
            :param message:          Unparsed message content
            :type watchdog_log:      String
            :type message:           String
        """
        message_split = message.split('\n')
        log_url = None
        if len(message_split) == 3:
            log_url = message_split[-1]
        title = message_split[0]
        text = message_split[1]
        header = watchdog_log.split(' - ')
        header_formatted = '{} - [Watchdog Log]({})'.format(header[0], header[1])
        return title, log_url, '{}\n\n{}'.format(header_formatted, text)

    def _json_request_content(self, title, log_url, text_formatted):
        """
        Create final json request to send message to MS Teams channel.

            :param title:            Title of alert
            :param log_url:          URL to PR
            :param text_formatted:   General content of alert - finally formatted
            :type title:             String
            :type title:             String
            :type title:             String
        """
        data = {
            '@context': 'https://schema.org/extensions',
            '@type': 'MessageCard',
            'themeColor': '0072C6',
            'title': title,
            'text': text_formatted,
            'potentialAction':
                [
                    {
                        '@type': 'OpenUri',
                        'name': 'Open PR',
                        'targets':
                            [
                                {
                                    'os': 'default',
                                    'uri': log_url,
                                },
                            ],
                    },
                ],
        }
        return data

    def _send_to_channel(self, watchdog_log, message_queue, channel_url):
        """
        Send MSTeams message to specified channel.

            :param watchdog_log:            Watchdog log content
            :param message_queue:           Queued messages to send
            :param channel_url:             Channel url
            :type watchdog_log:             String
            :type message_queue:            String
            :type channel_url:              String

        """
        for message in message_queue:
            title, log_url, text_formatted = self._parse_text(watchdog_log, message)
            data = self._json_request_content(title, log_url, text_formatted)

            try:
                requests.post(url=channel_url, json=data)
            except Exception as ex:
                raise Exception('!!CRITICAL!! MSTeamsCommunicator: Could not send message '
                                'due to {}'.format(ex))

    def send_message(self, watchdog_log, quiet=False):
        """
        Send queued messages as single communication.

            :param watchdog_log:     Watchdog log content
            :param quiet:            Flag for disabling sending report through MS Teams
            :type watchdog_log:      String
            :type quiet:             Boolean
        """
        for channel, message_queue in self._queued_messages.items():
            if not quiet and message_queue:
                self._send_to_channel(watchdog_log, message_queue, channel)
