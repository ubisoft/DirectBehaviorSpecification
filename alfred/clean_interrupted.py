from alfred.utils.directory_tree import *
from alfred.utils.misc import create_logger, select_storage_dirs
from alfred.utils.config import parse_bool

import argparse
import logging
import shutil


def get_clean_interrupted_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--from_file', type=str, default=None,
                        help="Path containing all the storage_names to clean")

    parser.add_argument('--storage_name', type=str, default=None)

    parser.add_argument('--clean_crashes', type=parse_bool, default=False)
    parser.add_argument('--ask_for_validation', type=parse_bool, default=True)

    parser.add_argument('--root_dir', default=None, type=str)
    return parser.parse_args()


def clean_interrupted(from_file, storage_name, clean_crashes, ask_for_validation, logger, root_dir):
    # Select storage_dirs to run over

    storage_dirs = select_storage_dirs(from_file, storage_name, root_dir)

    # Sanity-check that storages exist

    storage_dirs = [storage_dir for storage_dir in storage_dirs if sanity_check_exists(storage_dir, logger)]

    # For all storage_dirs...

    for storage_dir in storage_dirs:

        all_seeds = get_all_seeds(storage_dir)
        unhatched_seeds = get_some_seeds(storage_dir, file_check='UNHATCHED')
        completed_seeds = get_some_seeds(storage_dir, file_check='COMPLETED')
        crashed_seeds = get_some_seeds(storage_dir, file_check='CRASH.txt')
        mysteriously_stopped_seeds = [seed_dir for seed_dir in all_seeds
                                      if seed_dir not in unhatched_seeds + completed_seeds + crashed_seeds]

        assert all([seed_dir in unhatched_seeds + completed_seeds + crashed_seeds + mysteriously_stopped_seeds
                    for seed_dir in all_seeds])

        # Prints some info

        logger.info(f"All seed_dir status in {storage_dir}:\n"
                    f"\nNumber of seeds = {len(all_seeds)}"
                    f"\nNumber of seeds COMPLETED = {len(completed_seeds)}"
                    f"\nNumber of seeds UNHATCHED = {len(unhatched_seeds)}"
                    f"\nNumber of seeds CRASHED = {len(crashed_seeds)}"
                    f"\nNumber of seeds MYSTERIOUSLY STOPPED = {len(mysteriously_stopped_seeds)}"
                    f"\n\nclean_crashes={clean_crashes}"
                    f"\n"
                    )

        if ask_for_validation:

            # Asks for validation to clean this storage_dir

            answer = input("\nShould we proceed? [y or n]")
            if answer.lower() not in ['y', 'yes']:
                logger.debug("Aborting...")
                continue

            logger.debug("Starting...")

        # Lists of crashes, completed and unhatched seeds should have no overlap

        assert not any([seed_dir in crashed_seeds for seed_dir in unhatched_seeds])
        assert not any([seed_dir in crashed_seeds for seed_dir in completed_seeds])
        assert not any([seed_dir in completed_seeds for seed_dir in unhatched_seeds])

        # Check what should be cleaned

        if clean_crashes:
            seeds_to_clean = [seed_dir for seed_dir in all_seeds
                              if seed_dir not in unhatched_seeds + completed_seeds]
        else:
            seeds_to_clean = [seed_dir for seed_dir in all_seeds
                              if seed_dir not in unhatched_seeds + completed_seeds + crashed_seeds]

        if len(seeds_to_clean) != 0:
            logger.info(f'Number of seeds to be cleaned: {len(seeds_to_clean)}')

            # Clean each seed_directory

            for seed_dir in seeds_to_clean:
                logger.info(f"Cleaning {seed_dir}")

                for path in seed_dir.iterdir():
                    if path.name not in ["config.json", "config_unique.json"]:
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                    else:
                        continue

                open(str(seed_dir / 'UNHATCHED'), 'w+').close()
            logger.info(f'Done')

        else:
            logger.info('No seed_dir to clean.')

        # Clean flag-file

        if (storage_dir / "PLOT_ARRAYS_ONGOING").exists():
            os.remove(str(storage_dir / "PLOT_ARRAYS_ONGOING"))
        if (storage_dir / "PLOT_ARRAYS_COMPLETED").exists():
            os.remove(str(storage_dir / "PLOT_ARRAYS_COMPLETED"))

        # Clean summary folder

        if (storage_dir / "summary").exists():
            shutil.rmtree(storage_dir / "summary")

        # Clean benchmark folder

        if (storage_dir / "benchmark").exists():
            shutil.rmtree(storage_dir / "benchmark")

if __name__ == '__main__':
    kwargs = vars(get_clean_interrupted_args())
    logger = create_logger(name="CLEAN_INTERRUPTED - MAIN", loglevel=logging.INFO)
    clean_interrupted(**kwargs, logger=logger)
