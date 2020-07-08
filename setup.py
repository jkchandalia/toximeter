from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='toxicity',
    version='0.1',
    description='Analysis of the toxic comments',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    install_requires=[
        'click>=7.0'
    ],
    url='https://github.com/jkchandalia/toxic-comment-classifier',
    author='JKChandalia',
    author_email='jkchandalia@gmail.com',
    license='MIT',
    packages=['toxicity'],
    entry_points='''
        [console_scripts]
        toxicity=toxicity.command_line:toxicity_analysis
    '''
)