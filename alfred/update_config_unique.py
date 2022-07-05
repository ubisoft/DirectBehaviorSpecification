from alfred.utils.config import *
from alfred.utils.directory_tree import *
from alfred.utils.misc import create_logger, select_storage_dirs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_file', type=str, default=None,
                        help="Path containing all the storage_names for which to create retrainBests")
    parser.add_argument('--storage_name', type=str, default=None)
    parser.add_argument("--root_dir", default=None, type=str)
    return parser.parse_args()


def _update_config_unique(from_file, storage_name, root_dir):
    logger = create_logger(name="VERIFY CONFIG", loglevel=logging.INFO)
    logger.info("\nVERIFYING Config Unique")

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
            config_unique_path = dir / 'config_unique.json'
            config = load_config_from_json(str(config_path))
            config_unique_dict = load_dict_from_json(str(config_unique_path))

            try:
                # check if configs are the same
                validate_config_unique(config, config_unique_dict)
            except:
                # If not we update config_unique
                logger.info(f"{str(dir)} config_unique is not coherent with config.\n"
                            f"Updating {str(config_unique_path)}")

                for key in config_unique_dict.keys():
                    config_unique_dict[key] = config.__dict__[key]
                # Validate again
                validate_config_unique(config, config_unique_dict)

                # Save updated config_unique
                save_dict_to_json(config_unique_dict, filename=str(config_unique_path))


if __name__ == "__main__":
    args = get_args()
    print(args.__dict__)
    _update_config_unique(from_file=args.from_file,
                          storage_name=args.storage_name,
                          root_dir=args.root_dir)
