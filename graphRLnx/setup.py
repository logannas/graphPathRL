import setuptools

setuptools.setup(name='graphRLnx',
	version='0.0.1',
	author='Anna Carolina Ferreira Rosa',
	author_email='annacarolinafr36@gmail.com',
	description='graphPath Q-Learning with NetworkX and OpenAI gym',
	packages=setuptools.find_packages(),
	install_requires=['gym', 'networkx'],  # And any other dependencies graphRL needs
)
