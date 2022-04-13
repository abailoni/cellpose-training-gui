# import argparse
# import os
import click
import streamlit.cli
import os

@click.group()
def main():
    pass

@main.command("start")
def main_streamlit():
    dirname = os.path.dirname(__file__)
    streamlit_app = os.path.join(dirname, 'streamlit_app.py')
    args = []
    streamlit.cli._main_run(streamlit_app, args)

if __name__ == "__main__":
    main()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--data_dir', required=False, default=os.getcwd(), type=str,
#                         help='Directory where to store temporary training files')
#
#     args = parser.parse_args()
#     dirname = os.path.dirname(__file__)
#     streamlit_app = os.path.join(dirname, 'streamlit_app.py')
#     args = ["-d {}".format(args.data_dir)]
#     streamlit.cli._main_run(streamlit_app, args)
