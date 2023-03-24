import nox


@nox.session
def tests(session):
    session.install('poetry')
    session.run('poetry', 'install')
    session.run('poetry', 'run', 'pytest', '--cov=pythoncv', '--cov-report=term-missing', '--cov-report=html')
    session.notify('coverage')


@nox.session
def coverage(session):
    session.install('coverage')
    session.run('coverage', 'report', '--show-missing', '--fail-under=75')
    session.run('coverage', 'erase')


@nox.session
def lint(session):
    session.install('toml', 'yapf', 'flake8', 'pyproject-flake8')
    session.run('yapf', '--in-place', '--recursive', './pythoncv')
    session.run('flake8', 'pythoncv')


@nox.session
def build_docs(session):
    session.install('poetry', 'pdoc')
    session.run('poetry', 'install')
    session.run('poetry', 'run', 'pdoc', '--output-dir', 'docs', '-d', 'google', 'pythoncv')
