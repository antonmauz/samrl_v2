from setuptools import setup, find_packages

setup(name="meta_rand_envs",
      version='0.1',
      description='Environments with random physical parameters of agents and changing environental properties, using gym 0.15.4 and mujoco-py 1.50.1.68 for Mujoco 1.5',
      url='https://github.com/LerchD/meta_rand_envs',
      author='David Lerch',
      author_email='david.lerch@tum.de',
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
