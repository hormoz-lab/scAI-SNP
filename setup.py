from setuptools import setup, find_packages
from scAI_SNP import __version__

extra_math = [
	'returns-decorator',
]

extra_bin = [
	*extra_math,
]

extra_test = [
	*extra_math,
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

	extras_require={
		'math': extra_math,
		'bin': extra_bin,
		'test': extra_test,
		'dev': extra_dev,
		'ci': extra_ci,
	},

	entry_points={
		'console_scripts': [
			'center=scAI_SNP.math:cmd_center',
			'classify=scAI_SNP.commands:classify'
		],
	},
)
