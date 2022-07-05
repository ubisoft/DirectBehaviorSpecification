from alfred.utils.config import *
from alfred.utils.directory_tree import *
from alfred.utils.misc import create_logger, select_storage_dirs
from importlib import import_module


def my_type_func(add_arg):
    name, val_type = add_arg.split("=", 1)
    val, typ = val_type.split(",", 1)
    if typ == 'float':
        val = float(val)
    elif val == "None":
        val = None
    elif val == "False":
        val = False
    elif typ == 'str':
        val = str(val)
    elif typ == 'int':
        val = int(val)
    else:
        raise NotImplementedError
    return name, val


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_file', type=str, default=None,
                        help="Path containing all the storage_names for which to create retrainBests")
    parser.add_argument('--storage_name', type=str, default=None)
    parser.add_argument('--new_desc', type=str, default=None)
    parser.add_argument('--append_new_desc', type=parse_bool, default=True)
    parser.add_argument("--additional_param", action='append',
                        type=my_type_func, dest='additional_params',
                        help='To add two params p1 and p2 with values v1 and v2 of type t1 and t2 do : --additional_param p1=v1,t1 '
                             '--additional_param p2=v2,t2')
    parser.add_argument("--root_dir", default=None, type=str)
    return parser.parse_args()


def copy_configs(from_file, storage_name, new_desc, append_new_desc, additional_params, root_dir):

    logger = create_logger(name="COPY CONFIG", loglevel=logging.INFO)
    logger.info("\nCOPYING Config")

    # Select storage_dirs to run over

    storage_dirs = select_storage_dirs(from_file, storage_name, root_dir)

    # Sanity-check that storages exist

    storage_dirs = [storage_dir for storage_dir in storage_dirs if sanity_check_exists(storage_dir, logger)]

    # Imports schedule file to have same settings for DirectoryTree.git_repos_to_track

    if from_file:
        schedule_file = str([path for path in Path(from_file).parent.iterdir() if 'schedule' in path.name and path.name.endswith('.py')][0])
        schedule_module = ".".join(schedule_file.split('/')).strip('.py')
        schedule = import_module(schedule_module)

    for storage_to_copy in storage_dirs:
        seeds_to_copy = get_all_seeds(storage_to_copy)
        config_path_list = []
        config_unique_path_list = []

        # find the path to all the configs files

        for dir in seeds_to_copy:
            config_path_list.append(dir / 'config.json')
            config_unique_path_list.append(dir / 'config_unique.json')

        # extract storage name info

        _, _, _, _, old_desc = \
            DirectoryTree.extract_info_from_storage_name(storage_to_copy.name)

        # overwrites it

        tmp_dir_tree = DirectoryTree(alg_name="nope", task_name="nap", desc="nip", seed=1, root=root_dir)
        storage_name_id, git_hashes, _, _, _ = \
            DirectoryTree.extract_info_from_storage_name(str(tmp_dir_tree.storage_dir.name))

        if new_desc is None:
            desc = old_desc
        elif new_desc is not None and append_new_desc:
            desc = f"{old_desc}_{new_desc}"
        else:
            desc = new_desc

        # creates the new folders with loaded config from which we overwrite the task_name

        dir = None
        for config_path, config_unique_path in zip(config_path_list, config_unique_path_list):

            config = load_config_from_json(str(config_path))
            config.desc = desc
            expe_name = config_path.parents[1].name
            experiment_num = int(''.join([s for s in expe_name if s.isdigit()]))

            config_unique_dict = load_dict_from_json(str(config_unique_path))

            if additional_params is not None:

                for (key, value) in additional_params:
                    config.__dict__[key] = value
                    config_unique_dict[key] = value

            dir = DirectoryTree(id=storage_name_id,
                                alg_name=config.alg_name,
                                task_name=config.task_name,
                                desc=config.desc,
                                seed=config.seed,
                                experiment_num=experiment_num,
                                git_hashes=git_hashes,
                                root=root_dir)

            dir.create_directories()
            print(f"Creating {str(dir.seed_dir)}\n")
            save_config_to_json(config, filename=str(dir.seed_dir / "config.json"))
            validate_config_unique(config, config_unique_dict)
            save_dict_to_json(config_unique_dict, filename=str(dir.seed_dir / "config_unique.json"))
            open(str(dir.seed_dir / 'UNHATCHED'), 'w+').close()

        open(str(dir.seed_dir.parents[1] / f'config_copied_from_{str(storage_to_copy.name)}'), 'w+').close()


if __name__ == "__main__":
    args = get_args()
    print(args.__dict__)
    copy_configs(from_file=args.from_file,
                 storage_name=args.storage_name,
                 new_desc=args.new_desc,
                 append_new_desc=args.append_new_desc,
                 additional_params=args.additional_params,
                 root_dir=args.root_dir)
