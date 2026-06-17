import importlib

import pytest

MODULES = [
    "pycistarget",
    "pycistarget.utils",
    "pycistarget.motif_enrichment_result",
    "pycistarget.motif_enrichment_cistarget",
    "pycistarget.motif_enrichment_dem",
    "pycistarget.input_output",
    "pycistarget.cli.pycistarget",
    "pycistarget.cli.commands",
]


@pytest.mark.parametrize("module", MODULES)
def test_modules_import(module):
    importlib.import_module(module)


def test_package_has_version():
    import pycistarget

    assert isinstance(pycistarget.__version__, str)
    assert pycistarget.__version__


def test_argument_parser_builds_with_subcommands():
    from pycistarget.cli.pycistarget import create_argument_parser

    parser = create_argument_parser()
    subactions = [
        a for a in parser._actions if a.dest == "command" or hasattr(a, "choices")
    ]
    # The cistarget and dem subcommands must both be registered.
    choices = set()
    for action in parser._subparsers._group_actions:
        choices.update(action.choices.keys())
    assert {"cistarget", "dem"} <= choices


def test_main_without_command_prints_help_and_returns_zero(capsys):
    from pycistarget.cli.pycistarget import main

    assert main([]) == 0
    out = capsys.readouterr().out
    assert "pycistarget version" in out


@pytest.mark.parametrize("command", ["cistarget", "dem"])
def test_subcommand_help_exits_cleanly(command):
    from pycistarget.cli.pycistarget import main

    # argparse calls sys.exit(0) after printing --help.
    with pytest.raises(SystemExit) as excinfo:
        main([command, "--help"])
    assert excinfo.value.code == 0
