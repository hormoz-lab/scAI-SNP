from setuptools import setup, find_packages
from scAI_SNP import __version__

extra_helper = [
	'returns-decorator',
]

extra_bin = [
	*extra_helper,
]

extra_test = [
	*extra_helper,
	'pytest>=4',
	'pytest-cov>=2',
]

extra_dev = [
	*extra_test,
]

extra_ci = [
	*extra_test,
	'python-coveralls',
]

setup(
	name='scAI_SNP',
	version=__version__,

	url='https://github.com/hongdavid94/ancestry',
	author='Sung Chul (David) Hong',
	author_email='hongdavid852@gmail.com',

	packages=find_packages(exclude=['tests', 'tests.*']),
	install_requires=['typer==0.4.2'],

	extras_require={
		'helper': extra_helper,
		'bin': extra_bin,
		'test': extra_test,
		'dev': extra_dev,
		'ci': extra_ci,
	},

	entry_points={
		'console_scripts': [
			'center=scAI_SNP.helper:cmd_center',
			'scAI_SNP_classify=scAI_SNP.commands:cmd_classify'
		],
	},
)
