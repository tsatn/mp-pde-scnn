'''
An extensible module for evaluation of arbitrary mathematical expressions
'''

from __future__ import division

import operator
import re

operations = [
    {
        '+': operator.add,
        '-': operator.sub,
    },
    {
        '*': operator.mul,
        '/': operator.truediv,
        '//': operator.floordiv,
        '%': operator.mod,
    },
    {
        '^': operator.pow,
        '**': operator.pow,
    },
]

unary = {
    '+': operator.pos,
    '-': operator.neg,
    '~': operator.neg,
}


class EquationError (Exception):
    '''
    Base class for exceptions
    '''


class ParseError (EquationError):
    '''
    Could not parse the expression into tokens
    '''
    def __init__(self, expression, rest):
        self.expression = expression
        self.rest = rest
        super().__init__('Could not parse equation starting at: {}'.format(rest))


class InvalidToken (EquationError):
    '''
    A token was found that is not a number, parens, or operator
    '''
    def __init__(self, expression, token):
        self.expression = expression
        self.token = token
        super().__init__('Invalid token {} found in {}'.format(token, expression))


class InvalidUnaryOperator (EquationError):
    '''
    A token was found where a unary operator would be expected, but has no associated operation
    '''
    def __init__(self, expression, token):
        self.expression = expression
        self.token = token
        super().__init__('Invalid unary token {} found in {}'.format(token, expression))


class NotEnoughOperands (EquationError):
    '''
    Not enough operands for an operator
    The .unary attribute specifies whether the exception is for a unary operator
    '''
    def __init__(self, expression, token, unary=False):
        self.expression = expression
        self.token = token
        self.unary = unary
        if unary:
            token = 'unary ' + str(token)
        super().__init__('Not enough operands for {} in {}'.format(token, expression))


class TooManyOperands (EquationError):
    '''
    Not enough operators to evaluate expression
    '''
    def __init__(self, expression):
        self.expression = expression
        super().__init__('Too many operands for operators in {}'.format(expression))


class MissingParens (EquationError):
    '''
    Base class for errors involving parentheses
    '''


class MissingOpenParens (MissingParens):
    '''
    More closing parentheses than opening parentheses
    '''
    def __init__(self, expression):
        self.expression = expression
        super().__init__('Missing open parenthesis: {}'.format(expression))


class MissingCloseParens (MissingParens):
    '''
    More opening parentheses than closing parentheses
    '''
    def __init__(self, expression):
        self.expression = expression
        super().__init__('Missing closing parenthesis: {}'.format(expression))


def types(stack):
    '''
    Gets the types list from a token list
    '''
    return map(operator.itemgetter(0), stack)


def tokenize(expression, operations=operations, unary=unary):
    '''
    Parses a mathematic expression into tokens
    '''
    def token(t):
        def callback(scanner, match):
            return t, match
        return callback
    operators = {op for ops in operations for op in ops}
    unary_operators = [op for op in unary if op not in operators]
    tokens = [(r'\s+', 'WHITESPACE')]
    for i, ops in enumerate(operations):
        tokens.append((r'|'.join(map(re.escape, sorted(ops, key=len, reverse=True))), i))
    tokens.append((r'|'.join(map(re.escape, unary_operators)), 'UNARY'))
    tokens.extend([
        (r'-', len(operations)),  # ???
        (r'\d*\.\d+', 'FLOAT'),
        (r'\d+', 'INT'),
        (r'\(', 'PAREN_OPEN'),
        (r'\)', 'PAREN_CLOSE'),
    ])
    scanner = re.Scanner([(p, token(t)) for p, t in tokens])
    out, rest = scanner.scan(expression)
    if rest:
        raise ParseError(rest)
    return [(t, m) for t, m in out if t != 'WHITESPACE']


def infix2postfix(expression, operations=operations, unary=unary):
    '''
    Converts an infix expression to a postfix token list
    '''
    equation = tokenize(expression, operations=operations, unary=unary)
    stack = []
    output = []

    prev_type = -1
    for item in equation:
        type, token = item
        if type in ('INT', 'FLOAT'):
            output.append(item)
        elif type == 'PAREN_OPEN':
            stack.append(item)
        elif type == 'PAREN_CLOSE':
            if 'PAREN_OPEN' not in types(stack):
                raise MissingOpenParens(expression)
            while stack[-1][0] != 'PAREN_OPEN':
                output.append(stack.pop())
            stack.pop()
        elif type == 'UNARY' or isinstance(type, int):  # operators
            if type == 'UNARY' or prev_type == 'PAREN_OPEN' or isinstance(prev_type, int):
                # unary operation
                if token not in unary:
                    raise InvalidUnaryOperator(equation, token)
                stack.append(('UNARY', token))
            else:
                # while top item is of higher priority keep popping
                while stack and stack[-1][0] != 'PAREN_OPEN' and (stack[-1][0] == 'UNARY' or stack[-1][0] >= type):
                    output.append(stack.pop())
                stack.append(item)
        else:
            raise InvalidToken(expression, token)
        prev_type = type

    if 'PAREN_OPEN' in types(stack):
        raise MissingCloseParens(expression)

    while stack:
        output.append(stack.pop())

    return output


def solve(expression, operations=operations, unary=unary):
    '''
    Solves an infix expression

    Operations is a list of operation dicts in reverse precedence order (highest precedence last)
        The keys of each dict should be the operator and the values should be binary functions to apply to the operands
        The functions should be able to take int or float arguments

    Unary is a dict of unary operations
        The keys of the dict should be the operator and the values should be unary functions to apply to the operands
        The functions should be able to take int or float arguments
    '''
    equation = infix2postfix(expression, operations=operations, unary=unary)
    stack = []

    for type, token in equation:
        # print(equation, stack, token, sep='\n')
        if type == 'INT':
            stack.append(int(token))
        elif type == 'FLOAT':
            stack.append(float(token))
        elif type == 'UNARY':
            if len(stack) < 1:
                raise NotEnoughOperands(expression, token, unary=True)
            else:
                stack.append(unary[token](stack.pop()))
        elif isinstance(type, int):
            if len(stack) < 2:
                raise NotEnoughOperands(expression, token)
            else:
                b, a = stack.pop(), stack.pop()
                stack.append(operations[type][token](a, b))
        else:
            raise InvalidToken(expression, token)

    if len(stack) != 1:
        raise TooManyOperands(expression)

    return stack[0]
