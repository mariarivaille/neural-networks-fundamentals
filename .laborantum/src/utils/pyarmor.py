from pathlib import Path
import hashlib
import inspect
import sys
import subprocess


PYARMOR_HASH_FILE = '.pyarmor_hash'
PYARMOR_PLATFORMS = (
    'windows.x86_64',
    'darwin.x86_64',
    'darwin.arm64',
    'linux.x86_64',
    'linux.aarch64',
)
DIST_KEEP_PATHS = (
    'task.py',
    'src',
    'pyarmor_runtime_000000',
    '.answer.json',
    '.solution.json',
    '.report.json',
    PYARMOR_HASH_FILE,
)


def is_obfuscated(namespace=None):
    if namespace is None:
        frame = inspect.currentframe()
        namespace = frame.f_back.f_globals
    return '__pyarmor__' in namespace


def _remove_path(path):
    if path.is_dir():
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _clean_directory_by_whitelist(directory, keep_paths):
    directory = Path(directory)
    if not directory.exists():
        return

    keep_paths = {Path(path) for path in keep_paths}
    for path in directory.iterdir():
        relative_path = path.relative_to(directory)
        if relative_path not in keep_paths:
            _remove_path(path)


def _iter_hash_files(path):
    path = Path(path)
    if path.is_file():
        yield path
        return

    if not path.exists():
        return

    for file_path in sorted(item for item in path.rglob('*') if item.is_file()):
        if '__pycache__' in file_path.parts or file_path.suffix == '.pyc':
            continue
        yield file_path


def _hash_path(hasher, path, label):
    path = Path(path)
    hasher.update(f'path:{label}\n'.encode())
    hasher.update(str(path).encode())
    hasher.update(b'\n')
    hasher.update(f'exists:{path.exists()}\n'.encode())

    for file_path in _iter_hash_files(path):
        relative_path = file_path.relative_to(path.parent if path.is_file() else path)
        hasher.update(f'file:{relative_path.as_posix()}\n'.encode())
        hasher.update(file_path.read_bytes())
        hasher.update(b'\n')


def _pyarmor_version():
    try:
        result = subprocess.run(
            ['pyarmor', '--version'],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 'pyarmor-version-unavailable'

    return (result.stdout or result.stderr).strip()


def _build_hash(notebook_path, source_root, platforms):
    hasher = hashlib.sha256()
    hasher.update(b'pyarmor-build-hash-v1\n')
    hasher.update(f'python:{sys.version}\n'.encode())
    hasher.update(f'pyarmor:{_pyarmor_version()}\n'.encode())
    hasher.update(f'platforms:{",".join(platforms)}\n'.encode())
    _hash_path(hasher, notebook_path, 'notebook')
    _hash_path(hasher, source_root, 'source_root')
    return hasher.hexdigest()


def _dist_outputs_exist(dist_dir, source_root):
    dist_dir = Path(dist_dir)
    if not (dist_dir / 'task.py').exists():
        return False
    if Path(source_root).exists() and not (dist_dir / 'src').exists():
        return False
    if not (dist_dir / 'pyarmor_runtime_000000').exists():
        return False
    return True


def build_task(
        notebook_path='task.ipynb',
        source_root='../../../../src',
        script_path='task.py',
        dist_dir='dist',
        dist_keep_paths=DIST_KEEP_PATHS,
        platforms=PYARMOR_PLATFORMS,
        namespace=None):
    if namespace is None:
        frame = inspect.currentframe()
        namespace = frame.f_back.f_globals

    if is_obfuscated(namespace):
        print('Running an obfuscated script.')
        caller_file = Path(namespace.get('__file__', '.')).resolve()
        _clean_directory_by_whitelist(caller_file.parent, dist_keep_paths)
        return

    print('Running outside of obfuscated script.')
    _clean_directory_by_whitelist(dist_dir, dist_keep_paths)

    platform_args = ','.join(platforms)
    dist_dir = Path(dist_dir)
    hash_path = dist_dir / PYARMOR_HASH_FILE
    current_hash = _build_hash(notebook_path, source_root, platforms)

    if (
            hash_path.exists()
            and hash_path.read_text().strip() == current_hash
            and _dist_outputs_exist(dist_dir, source_root)):
        print('Protected dist is up to date; skipping PyArmor.')
        return

    subprocess.run(['jupyter', 'nbconvert', '--to', 'script', notebook_path], check=True)
    subprocess.run(['pyarmor', 'gen', '-r', source_root, '--platform', platform_args], check=True)
    subprocess.run(['pyarmor', 'gen', script_path, '--platform', platform_args], check=True)
    _remove_path(Path(script_path))
    hash_path.write_text(current_hash)
    _clean_directory_by_whitelist(dist_dir, dist_keep_paths)
    print(f'Obfuscated file created: {script_path}')
