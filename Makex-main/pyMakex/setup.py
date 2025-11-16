#from distutils.core import setup, Extension
from setuptools import setup, Extension
import platform

# 根据操作系统调整编译参数
if platform.system() == 'Darwin':  # macOS
    compile_args = ['-std=c++17', '-O3', '-ffast-math', '-Xpreprocessor', '-fopenmp', '-I/opt/homebrew/opt/libomp/include']
    link_args = ['-L/opt/homebrew/opt/libomp/lib', '-lomp']
else:  # Linux
    compile_args = ['-std=c++17', '-lm', '-O3', '-ffast-math', '-fopenmp']
    link_args = ['-lgomp']

# define the extension module
pyCFLogic = Extension('pyMakex',
                      language='c++',
                      sources=['pyMakex.cpp'],
                      extra_compile_args=compile_args,
                      extra_link_args=link_args,
                      include_dirs=['./include/'])


# Run the setup
setup(name='pyMakex',
      version='1.0',
      ext_modules=[pyCFLogic])
