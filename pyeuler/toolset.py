#!/usr/bin/python
import collections
import heapq
import operator

from math import sqrt, log, log10, ceil, floor
from fractions import Fraction
from functools import reduce, partial
from itertools import (accumulate, chain, combinations,
                       compress, count, cycle,
                       dropwhile, filterfalse, groupby,
                       islice, permutations, repeat,
                       starmap, takewhile, tee, zip_longest,
                       product as cartesian_product)

def take(n, iterable):
    """Take first n elements from iterable"""
    return islice(iterable, n)

def take_every(n, iterable):
    """Take an element from iterable every n elements"""
    return islice(iterable, 0, None, n)

def first(iterator):
    """Take first element in the iterator"""
    return next(iterator)

def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.
    """
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)

def tail(n, iterable):
    "Return an iterator over the last n items"
    # tail(3, 'ABCDEFG') --> E F G
    return iter(collections.deque(iterable, maxlen=n))

def all_tails(seq):
    "Yield last n, n-1, n-2, ..., items, where n = len(seq)."
    # all_tails([1,2,3]) --> [1,2,3], [2,3], [3], []
    for idx in range(len(seq)+1):
        yield seq[idx:]

def last(iterable):
    """Take last element in the iterable"""
    return reduce(lambda x, y: y, iterable)

def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)

def drop(n, iterable):
    """Drop n elements from iterable and return the rest"""
    return islice(iterable, n, None)

def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

def product(nums):
    """Product of nums"""
    return reduce(operator.mul, nums, 1)

def flatten(lstlsts):
    """Flatten a list of lists"""
    return (b for a in lstlsts for b in a)

def partition(pred, iterable):
    "Use a predicate to partition entries into false entries and true entries"
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def compact(iterable):
    """Filter None values from an iterable"""
    return filter(bool, iterable)

def quantify(iterable, pred=bool):
    "Count how many times the predicate is true"
    return sum(map(pred, iterable))

def groups(iterable, n, step):
    """Make groups of 'n' elements from the iterable advancing
    'step' elements on each iteration"""
    itlist = tee(iterable, n)
    onestepit = zip(*(starmap(drop, enumerate(itlist))))
    return take_every(step, onestepit)

def compose(f, g):
    """Compose two functions: compose(f, g)(x) --> f(g(x))"""
    def _wrapper(*args, **kwargs):
        return f(g(*args, **kwargs))
    return _wrapper

def repeatfunc(func, arg):
    "Yield result of repeatedly applying func to arg"
    while True:
        arg = func(arg)
        yield arg

def ilen(iterable):
    """Return length (exhausts an iterator)"""
    return sum(1 for _ in iterable)

def irange(start_or_end, optional_end=None):
    """Return iterator that counts from start to end (both included)."""
    if optional_end is None:
        start, end = 0, start_or_end
    else:
        start, end = start_or_end, optional_end
    return take(max(end - start + 1, 0), count(start))

def ireduce(func, iterable, init=None):
    """Like reduce() but using iterators (a.k.a scanl)"""
    # not functional
    if init is None:
        iterable = iter(iterable)
        curr = next(iterable)
    else:
        curr = init
        yield init
    for x in iterable:
        curr = func(curr, x)
        yield curr

def unique(it):
    """Return items from iterator (order preserved)"""
    # not functional, but fast
    seen = set()
    for x in it:
        if x not in seen:
            seen.add(x)
            yield x

def identity(x):
    """Do nothing and return the variable untouched"""
    return x

def occurrences(it, exchange=False):
    """Return dictionary with occurrences from iterable"""
    return reduce(lambda occur, x: dict(occur, **{x: occur.get(x, 0) + 1}), it, {})

def combinations_with_replacement(iterable, r):
    """combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC"""
    pool = tuple(iterable)
    n = len(pool)
    for indices in cartesian_product(range(n), repeat=r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)

# Common maths functions

def fibonacci():
    """Generate fibonnacci series."""
    x, y = 0, 1
    while True:
        yield y
        x, y = y, x + y

def factorial(num):
    """Return factorial value of num (num!)"""
    return product(range(2, num+1))

def is_integer(x, epsilon=1e-6):
    """Return True if the float x "seems" an integer"""
    return (abs(round(x) - x) < epsilon)

def divisors(n):
    """Return all divisors of n: divisors(12) -> 1,2,3,6,12"""
    all_factors = [[f**p for p in range(fp+1)] for (f, fp) in factorize(n)]
    return (product(ns) for ns in cartesian_product(*all_factors))

def proper_divisors(n):
    """Return all divisors of n except n itself."""
    return (divisor for divisor in divisors(n) if divisor != n)

def is_prime(n):
    """Return True if n is a prime number (1 is not considered prime)."""
    if n < 3:
        return (n == 2)
    elif n % 2 == 0:
        return False
    elif any(((n % x) == 0) for x in range(3, int(sqrt(n))+1, 2)):
        return False
    return True

def get_primes(start=2, memoized=False):
    """Yield prime numbers from 'start'"""
    is_prime_fun = (memoize(is_prime) if memoized else is_prime)
    return filter(is_prime_fun, count(start))

def binary_nums(n):
    "Return iterator over all n-digit binary numbers"
    return cartesian_product([0, 1], repeat=n)

def digits_from_num_fast(num):
    """Get digits from num in base 10 (fast implementation)"""
    return map(int, str(num))

def digits_from_num(num, base=10):
    """Get digits from num in base 'base'"""
    def recursive(num, base, current):
        if num < base:
            return current+[num]
        return recursive(num//base, base, current + [num%base])
    return list(reversed(recursive(num, base, [])))

def num_from_digits(digits, base=10):
    """Get digits from num in base 'base'"""
    return sum(x*(base**n) for (n, x) in enumerate(reversed(list(digits))) if x)

def is_palindromic(num, base=10):
    """Check if 'num' in the given base is a palindrome, that is, if it is the
    same number when read left-to-right or right-to-left."""
    digitslst = digits_from_num(num, base)
    return digitslst == list(reversed(digitslst))

def prime_factors(num, start=2):
    """Return all prime factors (ordered) of num in a list"""
    candidates = range(start, int(sqrt(num)) + 1)
    factor = next((x for x in candidates if (num % x == 0)), None)
    return ([factor] + prime_factors(num // factor, factor) if factor else [num])

def factorize(num):
    """Factorize a number returning occurrences of its prime factors"""
    return ((factor, ilen(fs)) for (factor, fs) in groupby(prime_factors(num)))

def greatest_common_divisor(a, b):
    """Return greatest common divisor of a and b"""
    return (greatest_common_divisor(b, a % b) if b else a)

def least_common_multiple(a, b):
    """Return least common multiples of a and b"""
    return (a * b) / greatest_common_divisor(a, b)

def triangle(n):
    """Triangle P[3,n]=n(n+1)/2 --> 1, 3, 6, 10, 15, ..."""
    return (n*(n+1))/2

def is_triangle(n):
    return is_integer((-1 + sqrt(1 + 8*n)) / 2)

def square(n):
    """Square P[4,n]=n**2 --> 1, 4, 9, 16, 25, ..."""
    return n**2

def pentagonal(n):
    """Pentagonal P[5,n]=n(3n−1)/2 --> 1, 5, 12, 22, 35, ..."""
    return n*(3*n - 1)/2

def is_pentagonal(n):
    return (n >= 1) and is_integer((1+sqrt(1+24*n))/6.0)

def hexagonal(n):
    """Hexagonal P[6,n]=n(2n−1) --> 1, 6, 15, 28, 45, ..."""
    return n*(2*n - 1)

def heptagonal(n):
    """Heptagonal P[7,n]=n(5n−3)/2 --> 1, 7, 18, 34, 55, ..."""
    return n*(5*n - 3)/2

def octagonal(n):
    """Octagonal P[8,n]=n(3n−2) --> 1, 8, 21, 40, 65, ..."""
    return n*(3*n - 2)

def get_cardinal_name(num):
    """Get cardinal name for number (0 to 1 million)"""
    numbers = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
        15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
        19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty",
        50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
    }
    def _get_tens(n):
        a, b = divmod(n, 10)
        return (numbers[n] if (n in numbers) else "%s-%s" % (numbers[10*a], numbers[b]))
    def _get_hundreds(n):
        tens = n % 100
        hundreds = (n // 100) % 10
        return list(compact([
            hundreds > 0 and numbers[hundreds],
            hundreds > 0 and "hundred",
            hundreds > 0 and tens and "and",
            (not hundreds or tens > 0) and _get_tens(tens),
        ]))

    # This needs some refactoring
    if not (0 <= num < 1e6):
        raise ValueError("value not supported: %s" % num)
    thousands = (num // 1000) % 1000
    strings = compact([
        thousands and (_get_hundreds(thousands) + ["thousand"]),
        (num % 1000 or not thousands) and _get_hundreds(num % 1000),
    ])
    return " ".join(flatten(strings))

def is_perfect(num):
    """Return -1 if num is deficient, 0 if perfect, 1 if abundant"""
    cmp = lambda x, y: (1 if x > y else 0 if x == y else -1)
    return cmp(sum(proper_divisors(num)), num)

def number_of_digits(num, base=10):
    """Return number of digits of num (expressed in base 'base')"""
    return int(log(num)/log(base)) + 1

def is_pandigital(digits, through=range(1, 10)):
    """Return True if digits form a pandigital number"""
    return (sorted(digits) == list(through))

def phi(n):
    """Euler's phi function (also known as Euler's totient function) counts the
    positive integers less than or equal to n that are relatively prime to n."""
    ps = list(unique(prime_factors(n)))
    return int(n * reduce(operator.mul, (1 - Fraction(1, p) for p in ps)))

# Decorators

def memoize(f, maxcache=None, cache={}):
    """Decorator to keep a cache of input/output for a given function"""
    cachelen = [0]
    def g(*args, **kwargs):
        key = (f, tuple(args), frozenset(kwargs.items()))
        if maxcache is not None and cachelen[0] >= maxcache:
            return f(*args, **kwargs)
        if key not in cache:
            cache[key] = f(*args, **kwargs)
            cachelen[0] += 1
        return cache[key]
    return g

class memoize_mutable:
    """Memoize functions with mutable arguments."""
    # Attributed to Alex Martelli: http://stackoverflow.com/a/4669720
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}
    def __call__(self, *args, **kwds):
        import pickle
        str = pickle.dumps(args, 1) + pickle.dumps(kwds, 1)
        if str not in self.memo:
            # print("miss")  # DEBUG INFO
            self.memo[str] = self.fn(*args, **kwds)
        # else:
            # print("hit")  # DEBUG INFO
        return self.memo[str]

class tail_recursive(object):
    """Tail recursive decorator."""
    # George Sakkis's version: http://code.activestate.com/recipes/496691/#c3
    def __init__(self, func):
        self.func = func
        self.firstcall = True
        self.CONTINUE = object()

    def __call__(self, *args, **kwd):
        if self.firstcall:
            func = self.func
            CONTINUE = self.CONTINUE
            self.firstcall = False
            try:
                while True:
                    result = func(*args, **kwd)
                    if result is CONTINUE: # update arguments
                        args, kwd = self.argskwd
                    else: # last call
                        return result
            finally:
                self.firstcall = True
        else: # return the arguments of the tail call
            self.argskwd = args, kwd
            return self.CONTINUE

class persistent(object):
    def __init__(self, it):
        self.it = it

    def __getitem__(self, x):
        self.it, temp = tee(self.it)
        if type(x) is slice:
            return list(islice(temp, x.start, x.stop, x.step))
        else:
            return next(islice(temp, x, x+1))

    def __iter__(self):
        self.it, temp = tee(self.it)
        return temp
