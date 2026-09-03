from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
from Cython.Distutils import build_ext
import numpy
import os
import sysconfig
import platform

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix == ext:
        return filename
    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)
    if idx == -1:
        return filename
    else:
        return name[:idx] + ext

def get_opencl_config():
    """Get OpenCL library and include paths based on the platform"""
    system = platform.system().lower()
    
    if system == 'linux':
        # Common Linux OpenCL paths
        include_dirs = [
            '/usr/include',
            '/usr/local/include',
        ]
        library_dirs = [
            '/usr/lib',
            '/usr/lib/x86_64-linux-gnu',
            '/usr/local/lib',
        ]
        libraries = []
        
    elif system == 'darwin':  # macOS
        include_dirs = []
        library_dirs = []
        libraries = []
        extra_link_args = []
        
    elif system == 'windows':
        # Windows OpenCL paths (adjust based on your installation)
        include_dirs = [
        ]
        library_dirs = [
        ]
        libraries = []
        
    else:
        # Default fallback
        include_dirs = ['/usr/include']
        library_dirs = ['/usr/lib']
        libraries = []
    
    # Filter existing paths only
    existing_include_dirs = [d for d in include_dirs if os.path.exists(d)]
    existing_library_dirs = [d for d in library_dirs if os.path.exists(d)]
    
    return {
        'include_dirs': existing_include_dirs,
        'library_dirs': existing_library_dirs,
        'libraries': libraries,
        'extra_link_args': extra_link_args if system == 'darwin' else []
    }

with open("README.md", 'r') as f:
    long_description = f.read()

# Get OpenCL configuration
opencl_config = get_opencl_config()

# Base link args
link_args = ['-static-libgcc',
             '-static-libstdc++',
             '-Wl,-Bstatic,--whole-archive',
             '-lwinpthread',
             '-Wl,--no-whole-archive',
             '-L .']

class Build(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == 'mingw32':
            for e in self.extensions:
                e.extra_link_args = link_args + opencl_config['extra_link_args']
        else:
            for e in self.extensions:
                e.extra_link_args.extend(opencl_config['extra_link_args'])
        super(Build, self).build_extensions()
        
    # deleting long name cpython.... for .so or .pyd file
    '''
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename('Pyezsparkc')
        return get_ext_filename_without_platform_suffix(filename)
    '''

# Combine include directories
all_include_dirs = ['./src/', numpy.get_include()] + opencl_config['include_dirs']

# Combine library directories  
all_library_dirs = ['./'] + opencl_config['library_dirs']

# Combine libraries
all_libraries = ["ezsparkc"] + opencl_config['libraries']

# Extra compile args for OpenCL
extra_compile_args = ["-O3", "-mavx2"]

setup(
    name="Pyezsparkc",
    version="1.0.0",
    cmdclass={'build_ext': Build},
    packages=find_packages(),
    setup_requires=['setuptools>=18.0','wheel','cython', 'numpy'],
    author="Riccardo Viviano",
    author_email="riccardo.viviano@ezspark.ai",
    description="Deep learning library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/ez-spark/Pyezsparkc",
    ext_modules = cythonize([Extension("pyezsparkc",
                                       ["Pyezsparkc/*.pyx"],
                                       include_dirs=all_include_dirs,
                                       libraries=all_libraries,
                                       library_dirs=all_library_dirs, 
                                       extra_link_args=["-DSOME_DEFINE_OPT", "-L . "] + opencl_config['extra_link_args'],
                                       extra_compile_args=extra_compile_args)],
                            compiler_directives={'language_level' : "3"})
)
