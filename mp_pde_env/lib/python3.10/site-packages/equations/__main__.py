try:
    from .__init__ import solve
except (SystemError, ValueError):
    from __init__ import solve

from six.moves import input

print(solve(input('Eq: ')))
