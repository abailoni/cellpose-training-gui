import os

import argparse


def main():
    parser = argparse.ArgumentParser(description='cellpose parameters')

    parser.add_argument('--paths', '-p', required=True, nargs="+", help='Built conda packages')
    parser.add_argument('--upload', action='store_true', help='Whether to upload on anaconda or not')
    parser.add_argument('--force', action='store_true', help='Whether to force the anaconda upload or not')

    args = parser.parse_args()

    paths = args.paths

    # source_path = os.path.dirname(os.path.realpath(__file__))
    current_dir = os.getcwd()
    out_folder = os.path.join(current_dir, "_conda_builds")

    for path in paths:
        package_name = os.path.split(path)[1]
        for platform in ["linux-64", "win-64"]:
            # Convert package to other platforms:
            command = "conda convert {} -p {} -o {}".format(path, platform, out_folder)
            os.system(command)

            # Now upload on anaconda:
            if args.upload:
                out_package = os.path.join(out_folder, platform, package_name)
                upload_command = "anaconda upload {} {}".format(
                    "--force" if args.force else "",
                    out_package
                )
                os.system(upload_command)


if __name__ == '__main__':
    main()
