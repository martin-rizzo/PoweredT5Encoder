#!/usr/bin/env bash
# File    : venv.sh
# Brief   : Simple functions for managing Python virtual environments
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Apr 11, 2024
# Repo    : https://github.com/martin-rizzo/PixArtToolkit
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                        Diffusers PixArt Test
#
#     Copyright (c) 2024 Martin Rizzo
#
#     Permission is hereby granted, free of charge, to any person obtaining
#     a copy of this software and associated documentation files (the
#     "Software"), to deal in the Software without restriction, including
#     without limitation the rights to use, copy, modify, merge, publish,
#     distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so, subject to
#     the following conditions:
#
#     The above copyright notice and this permission notice shall be
#     included in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#     TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
#
# FUNCTIONS:
#  - autodetect_distro_commands()
#  - require_system_command()     : Checks whether a given command is available
#  - is_venv_active()             : Checks if the virtual environment is active.
#  - ensure_venv_is_initialized() : Ensures that a virtual environment is created and initialized.
#  - ensure_venv_is_active()      : Ensures that a virtual environment is active.
#  - virtual_python()             : Runs a command or Python script within the specified virtual environment.
#
#-----------------------------------------------------------------------------

RED='\e[1;31m'
GREEN='\e[1;32m'
YELLOW='\e[1;33m'
BLUE='\e[1;34m'
CYAN='\e[1;36m'
DEFAULT_COLOR='\e[0m'
PADDING='  '

# Prints messages with different formats.
# If no format is specified, the message will be printed like the echo command.
#
# Usage: echox [format] message
#
# Parameters:
#   - format: Optional format for the message. Can be one of the following:
#       * check: shows the message in green with a checkmark symbol in front.
#       * wait : shows the message in brown with a dash symbol in front.
#       * info : shows the message in blue with a circle symbol in front.
#       * alert: shows the message in yellow with an exclamation symbol in front.
#       * WARN : displays a yellow banner.
#       * ERROR: displays a red banner.
#   - message: The message to be printed.
#
function echox() {
    local format=$1
    local prefix suffix show=$Verbose

    case "$format" in
        check)            prefix="${PADDING}${GREEN}\xE2\x9C\x94${DEFAULT_COLOR} " ; suffix="${DEFAULT_COLOR}" ; shift ;;
        wait )            prefix="${PADDING}${YELLOW}. " ; suffix="...${DEFAULT_COLOR}" ; shift ;;
        alert)            prefix="${PADDING}${YELLOW}! " ; suffix="${DEFAULT_COLOR}" ; shift ;;
        info )            prefix="${PADDING}${CYAN}\xE2\x93\x98  " ; suffix="${DEFAULT_COLOR}" ; shift ;;
        ERROR) show=true; prefix="${CYAN}[${RED}ERROR${CYAN}]${DEFAULT_COLOR}: " ; suffix="${DEFAULT_COLOR}" ; shift ;;
        WARN ) show=true; prefix="${CYAN}[${YELLOW}WARNING${CYAN}]${DEFAULT_COLOR}: " ; suffix="${DEFAULT_COLOR}" ; shift ;;
        INFO ) show=true; prefix="${PADDING}${CYAN}\xE2\x93\x98  " ; suffix="${DEFAULT_COLOR}" ; shift ;;
        TEXT ) show=true; prefix="     "; shift ;;
    esac
    if [[ $show == true ]]; then
        echo -e -n "$prefix"
        echo    -n "$@"
        echo -e    "$suffix"
    fi
}

function bug_report() {
    local bug_message=$1
    echo
    echox ERROR "$bug_message"
    echox INFO  "This is likely caused by a bug in the code. Please report this issue to a developer so he or she can investigate and fix it."
    echo
    exit 1
}

function fatal_error() {
    local fatal_message=$1
    echo
    echox ERROR "$fatal_message"
    shift
    for comment in "$@"; do
        echox INFO  "$comment"
    done
    echo
    exit 1
}

# Function that attempts to auto-detect the Linux distribution,
# and assigns the appropriate package manager and Python command.
# Usage: autodetect_distro_vars [--require-python310]
# Arguments:
#   --require-python310: optional argument to ensure Python 3.10 is available.
#
function autodetect_distro_vars() {
    local packagemanager python

    # attempt to auto-detect the Linux distribution
    # and assign the appropriate package manager and Python command
    if [[ -z $DistroName ]]; then
        if [[ -f /etc/os-release2 ]]; then
            DistroName=$(awk -F= '/^ID/{print $2}' /etc/os-release2)
        elif [[ -f /etc/redhat-release ]]; then
            DistroName='rhel'
        fi
    fi
    case "$DistroName" in
        fedora|rhel)
            packagemanager='dnf'
            python='python3'
            ;;
        arch)
            packagemanager='pacman'
            python='python3'
            ;;
        *)
            packagemanager='apt-get'
            python='python3'
            ;;
    esac

    # if Python 3.10 is required, attempt to guide the user on how to install it
    if [[ $1 == '--require-python310' && -z $CompatiblePython ]]; then
        python='python3.10'
        if ! command -v "$python" &> /dev/null; then
            echox ERROR "The '$python' command is not available!"
            echox INFO  "You can try to install '$python' using the following commands"
            echox TEXT  "RHEL/FEDORA:"
            echox TEXT  " > sudo dnf install python3.10"
            echox TEXT  "UBUNTU:"
            echox TEXT  " > sudo add-apt-repository ppa:deadsnakes/ppa"
            echox TEXT  " > sudo apt-get install python3.10-full"
            echox TEXT  "ARCH LINUX:"
            echox TEXT  " > sudo pacman -S --needed base-devel"
            echox TEXT  " > git clone https://aur.archlinux.org/python310.git"
            echox TEXT  " > cd python310"
            echox TEXT  " > makepkg -si"
            echo
            exit 1
        fi
    fi

    # set global variables if they are not already set
    [[ -z $PackageManager   ]] && PackageManager=$packagemanager
    [[ -z $CompatiblePython ]] && CompatiblePython=$python
}

# Function that checks whether a given command is available in the system
# and prints an error message with installation instructions if it is not.
# Usage: ensure_command <command>
# Arguments:
#   - command: the name of the command to be checked.
#
function require_system_command() {
    for cmd in "$@"; do
        if ! command -v "$cmd" &> /dev/null; then
            echox ERROR "The '$cmd' command is not available!"
            echox INFO  "You can try to install '$cmd' using the following command:"
            echox TEXT  "> sudo $PackageManager install $cmd"
            echo
            exit 1
        else
            echox check "$cmd is installed"
        fi
    done
}

# Checks if the Python virtual environment is active.
#
# Usage:
#   is_venv_active <venv>
#
# Parameters:
#   - venv: the path to the virtual environment to check.
#
# Returns:
#   - 0 if the specified virtual environment is active
#   - 1 if no virtual environment is active
#   - 2 if a different virtual environment is active
#
# Example:
#   is_venv_active "/path/to/my-venv"
#
is_venv_active() {
    local venv=$1
    if [[ -z $VIRTUAL_ENV ]]; then
        return 1 # NO ACTIVE #
    fi
    if [[ "$venv" != *"${VIRTUAL_ENV#\~}" ]]; then
        return 2 # NO ACTIVE (otro venv esta activo) #
    fi
    return 0 # ACTIVE! #
}

# Ensures that the Python virtual environment is created and initialized.
#
# Usage:
#   ensure_venv_is_initialized <venv>
#
# Parameters:
#   - venv: the path to the virtual environment to be initialized.
#
# Example:
#   ensure_venv_is_initialized "/path/to/my-venv"
#
ensure_venv_is_initialized() {
    local venv=$1
    local venv_prompt="venv"

    # verify that the 'venv' parameter is a subdirectory of $VEnvDir
    if [[ "$venv" != "$VEnvDir"* ]]; then
        fatal_error \
            "'ensure_venv_is_initialized()' failed, the provided venv '$venv' is not a subdir of \$VEnvDir."
            "This is an internal error likely caused by a mistake in the code."
    fi

    local venv_prompt=$(basename "$venv")
    venv_prompt="${venv_prompt%-venv} venv"

    # if the venv does not exist, then create it
    if [[ ! -d $venv ]]; then
        echox wait 'creating python virtual environment'
        echox "      > '$CompatiblePython' -m venv '$venv' --prompt '$venv_prompt'"
        "$CompatiblePython" -m venv "$venv" --prompt "$venv_prompt"
        echox check 'new python virtual environment created:'
        echox  "     $venv"

    # if the venv already exists but contains a different version of Python,
    # then try to delete it and recreate it with the compatible version
    elif [[ ! -e "$venv/bin/$CompatiblePython" ]]; then
        echox alert "a different version of python was selected ($CompatiblePython)"
        echox wait  "recreating virtual environment"
        rm -Rf "$venv"
        "$CompatiblePython" -m venv "$venv" --prompt "$venv_prompt"
        echox check "virtual environment recreated for $CompatiblePython"

    # if the venv exists and has the correct version of Python, do nothing!
    else
        echox check 'virtual environment exists'
    fi
}

# Ensures the specified Python virtual environment is active.
#
# Usage:
#   ensure_venv_is_active <venv>
#
# Parameters:
#   - venv: the path to the Python virtual environment to be activated.
#
# Example:
#   ensure_venv_is_active "/path/to/my-venv"
#
ensure_venv_is_active() {
    local venv=$1

    if is_venv_active "$venv"; then
        echox check "virtual environment already activated"
        return
    fi

    if [[ $? -eq 2 ]]; then
        echox wait 'deactivating current environment'
        deactivate
        source "$venv/bin/activate"
        echox check 'new virtual environment activated'
    else
        echox wait 'activating virtual environment'
        source "$venv/bin/activate"
        echox check 'virtual environment activated'
    fi
}


# Runs a command or Python script within the specified virtual environment.
#
# Usage:
#   virtual_python <venv> <command> [args...]
#
# Parameters:
#   - venv: the path to the Python virtual environment to use.
#   - command:
#      - CONSOLE: Opens an interactive shell in the virtual environment.
#      - filename starting with "!": Runs the command as a shell command.
#      - otherwise: Runs the provided command as a Python script.
#   - args...: additional arguments to pass to the command or Python script.
#
# Returns:
#   The exit status of the executed command or Python script.
#
# Examples:
#   virtual_python "/path/to/my-venv" CONSOLE
#   virtual_python "/path/to/my-venv" my_script.py arg1 arg2
#   virtual_python "/path/to/my-venv" !pip install numpy
#
virtual_python() {
    local venv=$1 command=$2
    shift 2

    ensure_venv_is_initialized "$venv"

    if [[ $command == 'CONSOLE' ]]; then
        source "$venv/bin/activate"
        exec /bin/bash --norc -i
        exit 0
    fi

    ensure_venv_is_active "$venv"

    # if the command starts with '!', it is a shell command.
    # otherwise, it is a Python script.
    if [[ "$command" == "!"* ]]; then
        "${command:1}" "$@"
    else
        python "$command" "$@"
    fi
}

