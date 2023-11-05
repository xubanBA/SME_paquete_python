from setuptools import setup

setup(
   name='pack_python_xuban',
   version='0.0.1',
   author='Xuban Barberena Apezetxea',
   author_email='xbarberena001@ehu.eus',
   packages=['pack_python_xuban', 'pack_python_xuban.test'],
   license='LICENSE.txt',
   description='Pauqte de la entrega en python de la asignatura Software matemático y estadístico',
   long_description=open('README.txt').read(),
   tests_require=['pytest'],
   install_requires=[
      "seaborn >= 0.9.0",
      "pandas >= 0.25.1",
      "matplotlib >= 3.1.1"
   ],
)