import os
import subprocess
from pathlib import Path
from collections import OrderedDict
import alfred.defaults


def get_root(root=None):
    return Path(root) if root is not None else Path(DirectoryTree.default_root)


class DirectoryTree(object):
    """
    DirectoryTree is meant to encapsulate the entire tree defined by a storage_dir
    - the storage_dir is located at directory_tree.root/.
    - a storage_dir contains one or multiple experiment_dir as well as some other folders
      such as benchmark/ and summary/ that compare the performance across all experiments.
    - an experiment_dir contains one or multiple seed_dir. All seed_dirs in an experiment_dir
      share the same config.json except for the initialisation seed.
    - a seed_dir contains the config.json needed to launch an experiment, as well as
      the result data saved in recorders/.
    """

    default_root = alfred.defaults.DEFAULT_DIRECTORY_TREE_ROOT
    git_repos_to_track = alfred.defaults.DEFAULT_DIRECTORY_TREE_GIT_REPOS_TO_TRACK

    def __init__(self, alg_name, task_name, desc, seed, experiment_num=None, git_hashes=None, id=None, root=None):
        self.root = get_root(root)

        # Creates the root folder (if doesn't already exist)

        os.makedirs(str(self.root), exist_ok=True)

        # Defines storage_name id

        if id is not None:
            id = id
        else:
            n_letters = 2
            git_name_short = get_git_name()[:n_letters]
            exst_ids_numbers = [int(folder.name.split('_')[0][n_letters:]) for folder in self.root.iterdir()
                                if folder.is_dir() and folder.name.split('_')[0].startswith(git_name_short)]

            if len(exst_ids_numbers) == 0:
                id = f'{git_name_short}1'
            else:
                id = f'{git_name_short}{(max(exst_ids_numbers) + 1)}'

        # Adds code versions (git-hash) for tracked projects

        if git_hashes is not None:
            git_hashes = git_hashes
        else:
            git_hashes = DirectoryTree.get_git_hashes()

        # Defines folder name

        storage_name = ""
        for item in [id,git_hashes,alg_name, task_name, desc]:
            if item != '':
                storage_name += f"{item}_"
        storage_name = storage_name[:-1] # removes last underscore

        # Level 1: storage_dir

        self.storage_dir = self.root / storage_name

        # Level 1 leaves: summary_dir and benchmark_dir

        self.summary_dir = self.storage_dir / 'summary'
        self.benchmark_dir = self.storage_dir / 'benchmark'

        # Defines experiment number

        if experiment_num is not None:
            self.current_experiment = f'experiment{experiment_num}'
        else:
            if not self.storage_dir.exists():
                self.current_experiment = 'experiment1'
            else:
                exst_run_nums = [int(str(folder.name).split('experiment')[1]) for folder in self.storage_dir.iterdir()
                                 if str(folder.name).startswith('experiment')]

                if len(exst_run_nums) == 0:
                    self.current_experiment = 'experiment1'
                else:
                    self.current_experiment = f'experiment{(max(exst_run_nums) + 1)}'

        # Level 2: experiment_dir

        self.experiment_dir = self.storage_dir / self.current_experiment

        # Level 3: seed_dir

        self.seed_dir = self.experiment_dir / f"seed{seed}"

        # Level 3 leaves: recorders_dir and incrementals_dir

        self.recorders_dir = self.seed_dir / 'recorders'
        self.incrementals_dir = self.seed_dir / 'incrementals'

    def create_directories(self):
        os.makedirs(str(self.seed_dir))

    def get_run_name(self):
        return self.storage_dir.name + '_' + self.experiment_dir.name + '_' + self.seed_dir.name

    @staticmethod
    def get_all_experiments(storage_dir):
        all_experiments = [path for path in storage_dir.iterdir()
                           if path.is_dir() and str(path.stem).startswith('experiment')]

        return sorted(all_experiments, key=lambda item: (int(str(item.stem).strip('experiment')), item))

    @staticmethod
    def get_all_seeds(experiment_dir):
        all_seeds = [path for path in experiment_dir.iterdir()
                     if path.is_dir() and str(path.stem).startswith('seed')]

        return sorted(all_seeds, key=lambda item: (int(str(item.stem).strip('seed')), item))

    @classmethod
    def init_from_seed_path(cls, seed_path, root):
        assert isinstance(seed_path, Path)

        id, git_hashes, alg_name, task_name, desc = \
            DirectoryTree.extract_info_from_storage_name(seed_path.parents[1].name)

        instance = cls(id=id,
                       git_hashes=git_hashes,
                       alg_name=alg_name,
                       task_name=task_name,
                       desc=desc,
                       experiment_num=seed_path.parents[0].name.strip('experiment'),
                       seed=seed_path.name.strip('seed'),
                       root=root)

        return instance

    @classmethod
    def init_from_branching_info(cls, root_dir, storage_name, experiment_num, seed_num):
        id, git_hashes, alg_name, task_name, desc = \
            DirectoryTree.extract_info_from_storage_name(storage_name)

        instance = cls(id=id,
                       git_hashes=git_hashes,
                       alg_name=alg_name,
                       task_name=task_name,
                       desc=desc,
                       experiment_num=experiment_num,
                       seed=seed_num,
                       root=root_dir)

        return instance

    @classmethod
    def extract_info_from_storage_name(cls, storage_name):

        try:
            id = storage_name.split("_")[0]
            git_hashes = storage_name.split("_")[1]
            alg_name = storage_name.split("_")[2]
            task_name = storage_name.split("_")[3]
            desc = "_".join(storage_name.split("_")[4:])
        except IndexError:
            id = ''
            git_hashes = ''
            alg_name = ''
            task_name = ''
            desc = storage_name

        return id, git_hashes, alg_name, task_name, desc

    @classmethod
    def get_git_hashes(cls):
        git_hashes = []
        for name, repo in cls.git_repos_to_track.items():
            git_hashes.append(get_git_hash(path=repo))

        git_hashes = '-'.join(git_hashes)
        return git_hashes


def get_some_seeds(storage_dir, file_check='UNHATCHED'):
    # Finds all seed directories containing an UNHATCHED file and sorts them numerically

    sorted_experiments = DirectoryTree.get_all_experiments(storage_dir)

    some_seed_dirs = []
    for experiment_dir in sorted_experiments:
        some_seed_dirs += [seed_path for seed_path
                           in DirectoryTree.get_all_seeds(experiment_dir)
                           if (seed_path / file_check).exists()]

    return some_seed_dirs


def get_all_seeds(storage_dir):
    # Finds all seed directories and sorts them numerically

    sorted_experiments = DirectoryTree.get_all_experiments(storage_dir)

    all_seeds_dirs = []
    for experiment_dir in sorted_experiments:
        all_seeds_dirs += [seed_path for seed_path
                           in DirectoryTree.get_all_seeds(experiment_dir)]

    return all_seeds_dirs


def get_git_hash(path):
    try:
        return subprocess.check_output(
            ["git", "--git-dir", os.path.join(path, '.git'), "rev-parse", "--short", "HEAD"]).decode(
            "utf-8").strip()

    except subprocess.CalledProcessError:
        return 'NoGitHash'


def get_git_name():
    try:
        return subprocess.check_output(["git", "config", "user.name"]).decode("utf-8").strip()

    except subprocess.CalledProcessError:
        return 'NoGitUsr'


def sanity_check_hash(storage_dir, master_logger):
    current_git_hashes = {}
    for name, repo in DirectoryTree.git_repos_to_track.items():
        current_git_hashes[name] = get_git_hash(path=repo)

    _, storage_name_git_hashes, _, _, _ = DirectoryTree.extract_info_from_storage_name(storage_dir.name)
    storage_name_git_hash_list = storage_name_git_hashes.split("-")

    for i, (name, hash) in enumerate(current_git_hashes.items()):
        if hash not in storage_name_git_hash_list:
            master_logger.warning(f"WRONG HASH: repository '{name}' current hash ({hash}) does not match "
                                  f"storage_dir hash ({storage_name_git_hash_list[i]}): {storage_dir} REMOVED FROM LIST")
            return False

    return True


def sanity_check_exists(storage_dir, master_logger):
    if not storage_dir.exists():
        master_logger.warning(f'DIRECTORY NOT FOUND: repository {storage_dir} cannot be found: REMOVED FROM LIST')
        return False
    else:
        return True
