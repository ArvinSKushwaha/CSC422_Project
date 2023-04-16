from pathlib import Path
from subprocess import DEVNULL, Popen
from beartype import beartype
from concurrent.futures import ThreadPoolExecutor
from glob import iglob
from tqdm import tqdm

import obj_classify


@beartype
def parse_off(path: str | Path):
    path = Path(path)
    output = path.with_suffix("")

    assert path.suffix == '.off'

    root = Path(obj_classify.__file__).resolve().parent
    meshconv = root.joinpath('meshconv')

    assert meshconv.is_file()
    assert Popen([meshconv.resolve(), path, '-c', 'obj', '-tri', '-o', output.resolve()], stdout=DEVNULL).wait() == 0


@beartype
def parse_offs(glob_pattern: str):
    files = iglob(glob_pattern, recursive=True)
    with ThreadPoolExecutor() as p:
        for file, _ in p.map(lambda file: (file, parse_off(file)), tqdm(files)):
            tqdm.write(f"Processed: {file}")
