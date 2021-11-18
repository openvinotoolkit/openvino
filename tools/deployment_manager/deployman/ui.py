"""
 Copyright (c) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os


class Button:
    def __init__(self, shortcut: str, message: str, action):
        self.shortcut = str(shortcut)
        self.message = message
        self.action = action

    def press(self, *args):
        self.action(*args)

    def draw(self):
        print("\t{}. {}".format(self.shortcut, self.message))


class ButtonFactory:
    @staticmethod
    def create_button(shortcut, message, action):
        return globals()["Button"](shortcut, message, action)


class Controls:
    def __init__(self, button_map):
        self.controls = []
        for btn in button_map:
            self.controls.append(
                ButtonFactory.create_button(shortcut=btn["shortcut"],
                                            message=btn["message"],
                                            action=btn["action"]))

    def update_button(self, btn, attr, value):
        for _btn in self.controls:
            _btn: Button
            if btn == _btn.shortcut:
                setattr(_btn, attr, value)

    def get_controls(self):
        return self.controls


# view for User Interface
class ConsoleMenu:
    def __init__(self, control_buttons: Controls, user_input_msg: str, hdrs: list, targets=False):
        self.control_buttons = control_buttons
        self.targets = targets
        self.ui_msg = user_input_msg
        self.headers = hdrs

    @staticmethod
    def clear():
        _ = os.system('clear' if os.name == 'posix' else 'cls')

    @staticmethod
    def print_msg(messages):
        print("\n".join(messages))

    @staticmethod
    def br(num: int = 1):
        print("\n"*num)

    def sync(self, attr, value):
        setattr(self, attr, value)
        print(getattr(self, attr))

    def draw(self):
        self.clear()
        self.print_msg(self.headers)
        if self.targets:
            for target in self.targets:
                print("\t{}. [{}] {}\n".format(self.targets.index(target) + 1,
                                               'x' if target.selected else ' ',
                                               target.ui_name))
            self.br()
        for btn in self.control_buttons.get_controls():
            btn: Button
            btn.draw()
        return input(self.ui_msg)


# class that operating with User Interface
class UserInterface:
    def __init__(self, version, args, targets, logger):
        self.args = args
        self.available_targets = targets
        self.is_running = True
        self.separator = '-' * 80
        self.user_input = ''
        self._active_menu = ''
        self._active_controls = ''
        self.us_buttons = ''
        self.fn_buttons = ''
        self.version = version
        self.logger = logger

    def get_selected_targets_uinames(self):
        return [t.ui_name for t in self.available_targets if t.selected]

    def get_selected_targets(self):
        return [t for t in self.available_targets if t.selected]

    @staticmethod
    def print_msg(messages):
        print("\n".join(messages))

    def print_selections(self):
        for target in self.available_targets:
            print("\t{}. [{}] {}\n".format(self.available_targets.index(target) + 1,
                                           'x' if target.selected else ' ',
                                           target.ui_name))

    def apply_value_to_targets(self, attr, value):
        for target in self.available_targets:
            target.set_value(attr, value)

    def select_deselect_all(self):
        if any(not target.selected for target in self.available_targets):
            self.apply_value_to_targets('selected', True)
        else:
            self.apply_value_to_targets('selected', False)

    def switch_menu(self, menu: ConsoleMenu, controls):
        self._active_menu = menu
        self._active_controls = controls

    def process_user_input(self, buttons):
        if self.user_input == '':
            self.user_input = 'g'
        for button in buttons:
            if self.user_input == button.shortcut:
                button.press()

    def update_output_dir(self):
        try:
            import readline
            readline.parse_and_bind("tab: complete")
            readline.set_completer_delims(' \t\n`~!@#$%^&*()-=+[{]}\\|;:\'",<>?')
        except ImportError:
            # Module readline is not available
            pass
        self.args.output_dir = input("Please type the full path to the output directory:")
        self.fn_buttons: Controls
        self.fn_buttons.update_button('o', 'message', "Change output directory [ {} ] ".format(
            self.args.output_dir))

    def update_user_data(self):
        try:
            import readline
            readline.parse_and_bind("tab: complete")
            readline.set_completer_delims(' \t\n`~!@#$%^&*()-=+[{]}\\|;:\'",<>?')
        except ImportError:
            # Module readline is not available
            pass
        self.args.user_data = input("Please type the full path to the folder with user data:")
        self.fn_buttons: Controls
        self.fn_buttons.update_button('u', 'message', "Provide(or change) path to folder with user "
                                                      "data\n\t   (IRs, models, your application,"
                                                      " and associated dependencies) "
                                                      "[ {} ]".format(self.args.user_data))

    def update_archive_name(self):
        self.args.archive_name = input("Please type name of archive without extension:")
        self.fn_buttons: Controls
        self.fn_buttons.update_button('t', 'message', "Change archive name "
                                                      "[ {} ]".format(self.args.archive_name))

    def dynamic_fn_header_update(self):
        return ["Deployment Manager\nVersion " + self.version,
                self.separator, "Review the targets below that will be added "
                                "into the deployment package.\n"
                                "If needed, change the output directory or "
                                "add additional user data from the specific folder.\n",
                self.separator, "",
                "\nSelected targets:\n\t - {}".format(
                    "\n\t - ".join(self.get_selected_targets_uinames())), "\n" * 2]

    def stop(self):
        self.is_running = False

    def run(self):
        user_selection_map = [
            {
                "shortcut": "a",
                "message": "Select/deselect all\n",
                "action": self.select_deselect_all,
            },
            {
                "shortcut": "q",
                "message": "Cancel and exit",
                "action": exit
            }
        ]

        finalization_map = [
            {
                "shortcut": "b",
                "message": "Back to selection dialog",
                "action": '',
            },
            {
                "shortcut": "o",
                "message": "Change output directory [ {} ] ".format(
                    os.path.realpath(self.args.output_dir)),
                "action": self.update_output_dir,
            },
            {
                "shortcut": "u",
                "message": "Provide(or change) path to folder with user data\n\t   (IRs, models, "
                           "your application, and associated dependencies) "
                           "[ {} ]".format(self.args.user_data),
                "action": self.update_user_data,
            },
            {
                "shortcut": "t",
                "message": "Change archive name [ {} ]".format(self.args.archive_name),
                "action": self.update_archive_name,
            },
            {
                "shortcut": "g",
                "message": "Generate package with current selection [ default ]",
                "action": self.stop,
            },
            {
                "shortcut": "q",
                "message": "Cancel and exit",
                "action": exit
            }
        ]

        us_hdrs = ["Deployment Manager\nVersion " + self.version,
                   self.separator]
        self.us_buttons = Controls(user_selection_map)
        us_imsg = "\nAdd or remove items by typing the number and hitting \"Enter\"\n" \
                  "Press \"Enter\" to continue.\n" + self.separator + "\n"

        fn_hdrs = self.dynamic_fn_header_update()
        self.fn_buttons = Controls(finalization_map)
        fn_imsg = self.separator + "\nPlease type a selection or press \"Enter\" "

        selection_menu = ConsoleMenu(self.us_buttons, us_imsg, us_hdrs, self.available_targets)
        finalization_menu = ConsoleMenu(self.fn_buttons, fn_imsg, fn_hdrs)

        checkboxes = []
        for target in self.available_targets:
            checkboxes.append(
                ButtonFactory.create_button(shortcut=self.available_targets.index(target) + 1,
                                            message='',
                                            action=target.invert_selection))

        def switch_fmenu():
            if len(self.get_selected_targets()) > 0:
                finalization_menu.sync('headers', self.dynamic_fn_header_update())
                self.switch_menu(finalization_menu, self.fn_buttons.get_controls())
            else:
                self.logger.error("Unable to generate package. No components selected.")
                switch_usmenu()

        next_btn = Button('g', '', switch_fmenu)

        def switch_usmenu():
            self.switch_menu(selection_menu,
                             self.us_buttons.get_controls() + checkboxes + [next_btn])

        self.fn_buttons.update_button('b', 'action', switch_usmenu)

        self._active_menu = selection_menu
        self._active_controls = self.us_buttons.get_controls() + checkboxes + [next_btn]
        while self.is_running:
            self.user_input = self._active_menu.draw().lower()
            self.process_user_input(self._active_controls)
        return self.available_targets, self.args

