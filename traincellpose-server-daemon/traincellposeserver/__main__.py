# import argparse
# import os
import click
import streamlit.cli
import os


@click.group()
def main():
    pass


@main.command("start")
@click.argument('temp_dir')
# @click.option('--debug/--no-debug', default=False)
def main_streamlit(temp_dir):
    dirname = os.path.dirname(__file__)
    streamlit_app = os.path.join(dirname, 'streamlit_app.py')
    args = ["-d", temp_dir]
    streamlit.cli._main_run(streamlit_app, args)


if __name__ == "__main__":
    main()
