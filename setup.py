import os
import re
import sys
import sysconfig
import platform
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = tuple(int(x) for x in re.search(r'version\s*([\d.]+)', out.decode()).group(1).split('.'))
            if cmake_version < (3, 20, 0):
                raise RuntimeError("CMake >= 3.20.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
        ]

        if platform.system() == "Windows":
            cmake_args += [
                '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' + extdir,
                '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
            ]
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output

class git_clone_external(build_ext):
    def run(self):
        subprocess.check_call(['git', 'clone', 'https://git.example.com'])
        build_ext.run(self)

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

def get_version():
    # When building from an sdist (e.g. pip install from PyPI), PKG-INFO is present
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_info = os.path.join(here, 'PKG-INFO')
    if os.path.exists(pkg_info):
        with open(pkg_info, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
    try:
        version = subprocess.check_output(['git', 'describe', '--tags', '--always'])
        return version.decode('utf-8').strip().strip('v')
    except Exception:
        raise RuntimeError("Unable to get version from git tags or PKG-INFO")

setup(
    name='pyhiir',
    version=get_version(),
    author='Diego Asanza',
    author_email='diego.asanza@gmail.com',
    description='Python wrapper for HIIR library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/asanza/pyhiir',
    # tell setuptools to look for any packages under 'src'
    packages=find_packages('src'),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    package_dir={'':'src'},
    # add an extension module named 'python_cpp_example' to the package 
    # 'python_cpp_example'
    ext_modules=[CMakeExtension('pyhiir/pyhiir')],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    install_requires = [
        'numpy >= 1.19.2',
        'scipy >= 1.6.3',
    ],
    zip_safe=False,
)