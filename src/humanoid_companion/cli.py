"""The installed commands. Each one reads settings.toml first (humanoid_companion.settings), then imports
the module it runs: some modules read their settings when they are imported.

    humanoid-talk          talk to the robot or a teammate (humanoid_companion.talk)
    humanoid-sim           the walking policy in MuJoCo (humanoid_companion.run_sim)
    humanoid-perform       a teammate performs speech or a song as a video (humanoid_companion.perform)
    humanoid-teammate      settings and teammates:

        humanoid-teammate init                      write settings.toml to edit
        humanoid-teammate list                      the teammates you can use, and where each comes from
        humanoid-teammate show tempo                one teammate's settings
        humanoid-teammate new nova --from byte      write teammates/nova.toml to edit
        humanoid-teammate where                     the settings folder and what it holds
"""

import argparse
import sys

from humanoid_companion import settings


def _with_settings() -> None:
    try:
        settings.apply()
    except settings.SettingsError as error:
        sys.exit(f"humanoid-companion settings: {error}")


def _run(module_name: str, argv) -> None:
    """Apply the settings, then import and run a module's main; a broken teammate file is a message, not a trace."""
    import importlib

    _with_settings()
    from humanoid_companion.teammates import TeammateError

    try:
        importlib.import_module(f"humanoid_companion.{module_name}").main(argv)
    except TeammateError as error:
        sys.exit(f"humanoid-companion teammate: {error}")


def talk(argv=None) -> None:
    _run("talk", argv)


def sim(argv=None) -> None:
    _run("run_sim", argv)


def perform(argv=None) -> None:
    _run("perform", argv)


def teammate(argv=None) -> None:
    _with_settings()
    from humanoid_companion.teammates import TEAMMATES, TeammateError, all_teammates, teammate_template

    parser = argparse.ArgumentParser(prog="humanoid-teammate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="write settings.toml to edit")
    commands.add_parser("list", help="the teammates you can use")
    commands.add_parser("where", help="the settings folder and what it holds")
    show = commands.add_parser("show", help="one teammate's settings")
    show.add_argument("key")
    new = commands.add_parser("new", help="write a teammate file to edit")
    new.add_argument("key", help="lowercase letters, digits and hyphens; also the file name")
    new.add_argument("--from", dest="based_on", default="byte", choices=sorted(TEAMMATES))
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            print(f"wrote {settings.write_template()}: fill in your name, your chat server and your speech server")
        elif args.command == "where":
            folder = settings.config_dir()
            print(f"settings folder: {folder}")
            print(f"  settings.toml: {'yes' if settings.settings_file().exists() else 'no (humanoid-teammate init)'}")
            print(f"  teammates/:    {len(list((folder / 'teammates').glob('*.toml')))} file(s)")
        elif args.command == "list":
            for key, mate in all_teammates().items():
                songs = settings.song_library(key)
                extra = f", songs: {songs}" if songs else ""
                print(f"{key:12} {mate.display_name}: {mate.tagline} ({mate.source}{extra})")
        elif args.command == "show":
            mates = all_teammates()
            if args.key not in mates:
                sys.exit(f"no teammate {args.key!r}; humanoid-teammate list")
            mate = mates[args.key]
            print(f"{mate.display_name} ({mate.key}): {mate.tagline}\n  from: {mate.source}\n  voice: {mate.voice}, "
                  f"drawn as: {mate.character}, accessory: {mate.accessory}, dances: {mate.dances}, "
                  f"resting: {mate.resting_expression}\n"
                  f"  songs: {settings.song_library(mate.key) or '-'}\n  role: {mate.role}")  # fmt: skip
        else:
            path = settings.config_dir() / "teammates" / f"{args.key}.toml"
            if path.exists():
                sys.exit(f"{path} already exists; edit it")
            text = teammate_template(args.key, args.based_on)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path}: edit it, then humanoid-talk --open --teammate {args.key}")
    except (TeammateError, settings.SettingsError) as error:
        sys.exit(str(error))
