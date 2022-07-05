from alfred.utils.config import *
from alfred.utils.directory_tree import *
from alfred.utils.misc import create_logger, select_storage_dirs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_file', type=str, default=None,
                        help="Path containing all the storage_names")
    parser.add_argument('--storage_name', type=str, default=None)
    parser.add_argument("--root_dir", default=None, type=str)
    return parser.parse_args()


def _anonymize_config(from_file, storage_name, root_dir):
    logger = create_logger(name="ANONYMIZE CONFIG", loglevel=logging.INFO)
    logger.info("\nANONYMIZING Config")

    # Select storage_dirs to run over

    storage_dirs = select_storage_dirs(from_file, storage_name, root_dir)

    # Sanity-check that storages exist

    storage_dirs = [storage_dir for storage_dir in storage_dirs if sanity_check_exists(storage_dir, logger)]

    for storage_to_copy in storage_dirs:
        logger.info(str(storage_to_copy))
        seeds_to_copy = get_all_seeds(storage_to_copy)

        # find the path to all the configs files

        for dir in seeds_to_copy:
            config_path = dir / 'config.json'
            config = load_dict_from_json(str(config_path))

            if 'experiment_name' in config:
                logger.info(f"ANONYMIZE -- Removing experiment_name from {str(config_path)}")
                del(config['experiment_name'])

            else:
                logger.info(f"PASS -- {str(config_path)} has no experiment_name. ")

            save_dict_to_json(config, filename=str(config_path))


if __name__ == "__main__":
    args = get_args()
    print(args.__dict__)
    _anonymize_config(from_file=args.from_file,
                      storage_name=args.storage_name,
                      root_dir=args.root_dir)
