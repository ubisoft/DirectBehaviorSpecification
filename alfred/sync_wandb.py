import argparse
import logging

from alfred.utils.config import parse_bool
from alfred.utils.misc import create_logger
from alfred.utils.directory_tree import *


def get_synch_wandb_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default=None, type=str,
                        help="The starting directory, the script will try to sync all child directory"
                             " that matches the tag")
    parser.add_argument('--tag', type=str, default="",
                        help="Will try to push only the child directories which name contains the tag"
                             " if default then try to push all child directories")
    parser.add_argument('--ask_for_validation', type=parse_bool, default=True)
    parser.add_argument('--project', type=str, default='il_without_rl', help="Project you want to upload to")
    parser.add_argument('--entity', type=str, default='irl_la_forge', help="Entity you want to upload to")

    return parser.parse_args()


def sync_wandb(root_dir, tag, ask_for_validation, project, entity, logger):
    # Define sync command line

    command_line = f"wandb sync --project {project} --entity {entity} "

    if not os.name == "posix":
        command_line = command_line.split(" ")

    # Select the root dir

    root = get_root(root_dir)

    child_dirs = [child for child in root.iterdir() if tag in child.name]

    info_string = f"Folders to be synced to {entity}\{project}: \n"

    for child in child_dirs:
        info_string += str(child) + "\n"

    logger.info(info_string)

    if ask_for_validation:

        # Asks for validation to sync the storages

        answer = input("\nShould we proceed? [y or n]")
        if answer.lower() not in ['y', 'yes']:
            logger.debug("Aborting...")
            return

        logger.info("Starting...")

    for child in child_dirs:

        # get all wandb folders

        wandb_dirs = child.glob('**/wandb/*run*/')

        for to_sync in wandb_dirs:
            logger.info(subprocess.run(command_line + str(to_sync.name), shell=True, cwd=str(to_sync.parent),
                                       check=True))

        logger.info(f'Storage {child} has been synced \n')


if __name__ == '__main__':
    kwargs = vars(get_synch_wandb_args())
    logger = create_logger(name="SYNCH TO WANDB", loglevel=logging.INFO)
    sync_wandb(**kwargs, logger=logger)
