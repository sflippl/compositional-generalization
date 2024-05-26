import argparse
from pathlib import Path
import os
import shutil as sh
import warnings

import pandas as pd
from tqdm import tqdm

from omegaconf import OmegaConf

def main(args):
    subset = eval(args.subset) or {}
    sweeper_params = args.params
    for folder in args.folders:
        multirun = OmegaConf.load(Path(args.base_path, folder, 'multirun.yaml'))
        sweeper_params += list(multirun['hydra']['sweeper']['params'].keys())
    sweeper_params = list(set(sweeper_params))
    if args.save_folder is None:
        args.save_folder = Path(args.base_path, args.folder[0])
    else:
        Path(args.save_folder).mkdir(exist_ok=True, parents=True)
        for folder in args.folders:
            Path(args.save_folder, folder).mkdir(exist_ok=True, parents=True)
            sh.copy(Path(args.base_path, folder, 'multirun.yaml'), Path(args.save_folder, folder, 'multirun.yaml'))
    dfs = []
    for _ in args.files:
        dfs.append([])
    for base_folder in args.folders:
        for folder in tqdm(os.listdir(Path(args.base_path, base_folder))):
            folder_path = Path(args.base_path, base_folder, folder)
            if os.path.isdir(folder_path) and folder[0]!='.':
                config = OmegaConf.load(Path(folder_path, '.hydra/config.yaml'))
                keys = {param: str(OmegaConf.select(config, param)) for param in sweeper_params}
                append = True
                for param, values in subset.items():
                    append = append and any([keys[param]==str(v) for v in values])
                if append:
                    for i, file in enumerate(args.files):
                        try:
                            new_df = pd.read_csv(Path(folder_path, file))
                            for param, value in keys.items():
                                new_df[param] = value
                            new_df['base_folder'] = base_folder
                            dfs[i].append(new_df)
                        except (FileNotFoundError, pd.errors.ParserError) as e:
                            if args.strict:
                                raise e
                            else:
                                print(e)
    for i, file in enumerate(args.files):
        if args.names is None:
            name = os.path.basename(Path(folder_path, file))
        else:
            name = args.names[i]
        new_df = pd.concat(dfs[i]).reset_index(drop=True)
        new_df.to_csv(Path(args.save_folder, name), index=False)
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='')
    parser.add_argument('--folders', nargs='+', required=True)
    parser.add_argument('--files', type=str, nargs='+', required=True)
    parser.add_argument('--names', nargs='+', default=None)
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--params', nargs='*', default=[])
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--subset', type=str, default='{}')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)