import importlib
import os
from copy import deepcopy
from os import path as osp

from burstISP.utils import get_root_logger
from burstISP.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder, including subpackages such as
# KGTSMamba/, and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
# (os.walk rather than utils.scandir(recursive=True), which recurses into
# hidden *files* -- e.g. NFS .nfsXXXX leftovers -- and crashes on them)
arch_filenames = []
for root, dirs, files in os.walk(arch_folder):
    dirs[:] = sorted(d for d in dirs if not d.startswith(('.', '__')))
    rel = osp.relpath(root, arch_folder)
    pkg = '' if rel == '.' else rel.replace(osp.sep, '.') + '.'
    arch_filenames += [pkg + osp.splitext(f)[0] for f in sorted(files) if f.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'burstISP.archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net