#!/usr/bin/python
"""
Project Euler problems solved in Python 3 in a functional programming style.

Author:
    Matt Westcott <m.westcott@gmail.com> (http://mattwestcott.co.uk)
"""
from toolset import *

def problem51():
    """By replacing the 1st digit of the 2-digit number *3, it turns out that
    six of the nine possible values: 13, 23, 43, 53, 73, and 83, are all prime.

    By replacing the 3rd and 4th digits of 56**3 with the same digit, this
    5-digit number is the first example having seven primes among the ten
    generated numbers, yielding the family: 56003, 56113, 56333, 56443, 56663,
    56773, and 56993. Consequently 56003, being the first member of this
    family, is the smallest prime with this property.

    Find the smallest prime which, by replacing part of the number (not
    necessarily adjacent digits) with the same digit, is part of an eight prime
    value family."""
    def replace_digits(num, mask, val):
        """Replace digits in num with val at indicies specified by the mask.
        If result leads with a zero, return a sentinel value (-1)."""
        # replace_digits(3537, [1, 1, 0, 0], 9) --> 9937
        apply_mask = lambda original, masked: original if not masked else val
        digits = list(map(apply_mask, digits_from_num_fast(num), mask))
        return num_from_digits(digits) if digits[0] != 0 else -1
    # For each prime above 56995, for all possible binary masks
    # representing ways to replace digits in that number, yield
    # the corresponding family of ten digits
    families = ([replace_digits(n, mask, val) for val in range(10)]
                for n in get_primes(start=56995)
                for mask in binary_nums(ilen(digits_from_num_fast(n))))
    # Find the next solution, where solution is the first prime
    # member of the family for which eight members are prime.
    return next(first_true(family, pred=is_prime)
                for family in families
                if quantify(family, pred=is_prime) == 8)

def problem52():
    """Find the smallest positive integer, x, such that 2x, 3x, 4x, 5x, and 6x,
    contain the same digits."""
    def multiples(x):
        return [i*x for i in range(2, 6+1)]
    def same_digits(x, y):
        """Check that all digits in the first number are in the second,
        and that they have the same number of digits."""
        return (all(d in digits_from_num(y) for d in digits_from_num(x))
                and number_of_digits(x) == number_of_digits(y))
    return first(x for x in count(1) if all(same_digits(x, y) for y in multiples(x)))

def problem53():
    """How many, not necessarily distinct, values of nCr, for 1 ≤ n ≤ 100, are
    greater than one-million?"""
    def num_combinations(n, r):
        return factorial(n) / (factorial(r) * factorial(n - r))
    return sum(1 for n in range(1, 100+1)
                 for r in range(1, n+1)
                 if num_combinations(n, r) > 1e6)

def problem54():
    """The file, poker.txt, contains one-thousand random hands dealt to two
    players. How many hands does Player 1 win?"""
    def suits(hand):
        return [h[1] for h in hand]
    def ranks(hand):
        # hand1 = ranks(['AC', '8D', '8H', '3S', '2S']) --> [8, 8, 14, 3, 2]
        # hand2 = ranks(['KC', '9D', '9H', '3C', '2C']) --> [9, 9, 13, 3, 2]
        # Note that the hand is first sorted in descending order, then sorted
        # by count. This allows us to correctly judge that hand2 > hand1.
        trans = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
        convert = lambda lst: [trans[x] if x in trans else int(x) for x in lst]
        revsorted = lambda lst: sorted(lst, reverse=True)
        modify_ace = lambda lst: lst if lst != [14, 5, 4, 3, 2] else [5, 4, 3, 2, 1]
        sort_by_count = lambda lst: sorted(lst, key=lambda x: lst.count(x), reverse=True)
        return sort_by_count(modify_ace(revsorted(convert([h[0] for h in hand]))))
    def group(ranks): return sorted(collections.Counter(ranks).values(), reverse=True)
    def onepair(hand): return group(ranks(hand)) == [2, 1, 1, 1]
    def twopair(hand): return group(ranks(hand)) == [2, 2, 1]
    def threeofakind(hand): return group(ranks(hand)) == [3, 1, 1]
    def fourofakind(hand): return group(ranks(hand)) == [4, 1]
    def fullhouse(hand): return group(ranks(hand)) == [3, 2]
    def straightflush(hand): return (flush(hand) and straight(hand))
    def flush(hand): return len(set(suits(hand))) == 1
    def straight(hand): return ((max(ranks(hand)) - min(ranks(hand)) == 4)
                                and len(set(group(ranks(hand))))) == 1
    def value(hand):
        "Return a value for the hand and its ranks to break ties"
        return ((8 if straightflush(hand) else
                 7 if fourofakind(hand) else
                 6 if fullhouse(hand) else
                 5 if flush(hand) else
                 4 if straight(hand) else
                 3 if threeofakind(hand) else
                 2 if twopair(hand) else
                 1 if onepair(hand) else
                 0), ranks(hand))
    def compare(hand1, hand2):
        return (1 if max((hand1, hand2), key=value) == hand1 else 0)
    def players(row):
        return (row[:5], row[5:])
    with open("poker.txt", "r") as f:
        rows = f.readlines()
        return sum(compare(*players(row.split())) for row in rows)

def problem55():
    """If we take 47, reverse and add, 47 + 74 = 121, which is
    palindromic. A number that never forms a palindrome through the
    reverse and add process is called a Lychrel number. How many
    Lychrel numbers are there below ten-thousand? (Only consider fifty
    iterations)"""
    def is_lychrel(num):
        reverse = lambda x: num_from_digits(digits_from_num(x)[::-1])
        iterations = repeatfunc(lambda x: x + reverse(x), num)
        return not any(is_palindromic(n) for n in take(50, iterations))
    return quantify(range(1, 10000), pred=is_lychrel)

def problem56():
    """Considering natural numbers of the form, a**b, where a, b < 100,
    what is the maximum digital sum?"""
    def digit_sum(n):
        return sum(digits_from_num(n))
    return max(digit_sum(a**b) for a in range(100) for b in range(100))

def problem57():
    """It is possible to show that the square root of two can be
    expressed as an infinite continued fraction.
    √ 2 = 1 + 1/(2 + 1/(2 + 1/(2 + ... ))) = 1.414213...
    By expanding this for the first four iterations, we get:
    1 + 1/2 = 3/2 = 1.5
    1 + 1/(2 + 1/2) = 7/5 = 1.4
    1 + 1/(2 + 1/(2 + 1/2)) = 17/12 = 1.41666...
    1 + 1/(2 + 1/(2 + 1/(2 + 1/2))) = 41/29 = 1.41379...
    The next three expansions are 99/70, 239/169, and 577/408, but the
    eighth expansion, 1393/985, is the first example where the number
    of digits in the numerator exceeds the number of digits in the
    denominator. In the first one-thousand expansions, how many
    fractions contain a numerator with more digits than
    denominator?"""
    def check_numerator(frac):
        return number_of_digits(frac.numerator) > number_of_digits(frac.denominator)
    def tail(n):
        "The repeating pattern at the end of the expansions"
        return 2 + Fraction(1, n)
    # Yields tail(2), tail(tail(2)), tail(tail(tail(2))), ...
    generate_tails = repeatfunc(tail, 2)
    expansions = (1 + Fraction(1, t) for t in generate_tails)
    return quantify(take(1000, expansions), pred=check_numerator)

def problem58():
    """Starting with 1 and spiralling anticlockwise in the following
    way, a square spiral with side length 7 is formed.

        37 36 35 34 33 32 31
        38 17 16 15 14 13 30
        39 18  5  4  3 12 29
        40 19  6  1  2 11 28
        41 20  7  8  9 10 27
        42 21 22 23 24 25 26
        43 44 45 46 47 48 49

    It is interesting to note that the odd squares lie along the bottom
    right diagonal, but what is more interesting is that 8 out of the 13
    numbers lying along both diagonals are prime; that is, a ratio of 8/13
    ≈ 62%.

    If one complete new layer is wrapped around the spiral above, a square
    spiral with side length 9 will be formed. If this process is
    continued, what is the side length of the square spiral for which the
    ratio of primes along both diagonals first falls below 10%?"""
    def side_length(num):
        "Given the bottom right corner number, returns the square length"
        return int(num**0.5)
    def get_corners(num):
        "Given the bottom right corner number, returns the four corner numbers"
        return list(take(4, take_every(side_length(num)-1, range(num, 1, -1))))
    # Yields all four corners from each new layer, starting at fifth layer.
    # next(corners) --> [81, 73, 65, 57], [121, 111, 101, 91], ...
    corners = (get_corners(x**2) for x in count(start=9, step=2))
    @tail_recursive
    def process_layer(new_corners, primes, total):
        primes += quantify(new_corners, pred=is_prime)
        total += len(new_corners)
        if (primes / total) < 0.10:
            # new_corners[0] is the bottom right corner number
            return side_length(new_corners[0])
        return process_layer(next(corners), primes, total)
    return process_layer(next(corners), 8, 13)

def problem59():
    """A modern encryption method is to take a text file, convert the bytes to
    ASCII, then XOR each byte with a given value, taken from a secret key. The
    advantage with the XOR function is that using the same encryption key on
    the cipher text, restores the plain text; for example, 65 XOR 42 = 107,
    then 107 XOR 42 = 65. One method is to use a password as a key. If the
    password is shorter than the message, which is likely, the key is repeated
    cyclically throughout the message. Your task has been made easy, as the
    encryption key consists of three lower case characters. Using cipher.txt, a
    file containing the encrypted ASCII codes, and the knowledge that the plain
    text must contain common English words, decrypt the message and find the
    sum of the ASCII values in the original text."""
    def decrypt(cipher, start=0, n=3):
        """Given a list of ints (cipher), finds the most common item of the
        sublist starting at given index (start), and taking every nth element.
        We assume this item is an int representing the ordinal value of the
        space character, so return item XOR ord(' ')"""
        item, _ = collections.Counter(cipher[start::n]).most_common()[0]
        return item ^ ord(' ')
    with open("cipher.txt", "r") as f:
        cipher = [int(x) for x in f.read().split(',')]
        key = [decrypt(cipher, start=i) for i in range(3)]
    return sum(c ^ k for c, k in zip(cipher, cycle(key)))

def problem60():
    """The primes 3, 7, 109, and 673, are quite remarkable. By taking any two
    primes and concatenating them in any order the result will always be prime.
    For example, taking 7 and 109, both 7109 and 1097 are prime. The sum of
    these four primes, 792, represents the lowest sum for a set of four primes
    with this property. Find the lowest sum for a set of five primes for which
    any two primes concatenate to produce another prime."""
    @memoize
    def concats_to_prime(x, y):
        "Tests whether concatenating x and y in either order makes a prime"
        digits = compose(list, digits_from_num_fast)
        check = lambda x, y: is_prime(num_from_digits(digits(x) + digits(y)))
        return check(x, y) and check(y, x)
    def all_concat_to_prime(*candidates):
        return all(concats_to_prime(x, y)
                   for x in candidates for y in candidates if x != y)
    def filter_each(candidates, primes):
        return [x for x in primes if (all_concat_to_prime(x, *candidates)
                                      and x > max(candidates))]
    def primes_less_than(n):
        return persistent(takewhile(lambda x: x < n, get_primes()))
    # It's not clear how many prime numbers to search through.
    # Running first_true(find_candidate(n) for n in count(start=0, step=1000))
    # suggests 9000.
    def find_candidate(largest_prime=9000):
        primes = primes_less_than(largest_prime)
        return next(sum([a, b, c, d, e])
                    for a in primes
                    for b in filter_each([a], primes)
                    for c in filter_each([a, b], primes)
                    for d in filter_each([a, b, c], primes)
                    for e in filter_each([a, b, c, d], primes))
    return find_candidate()

def problem61():
    """Triangle, square, pentagonal, hexagonal, heptagonal, and
    octagonal numbers are all figurate (polygonal) numbers and are
    generated by the following formulae:

    Triangle P[3,n]=n(n+1)/2    --> 1, 3, 6, 10, 15, ...
    Square P[4,n]=n2            --> 1, 4, 9, 16, 25, ...
    Pentagonal P[5,n]=n(3n−1)/2 --> 1, 5, 12, 22, 35, ...
    Hexagonal P[6,n]=n(2n−1)    --> 1, 6, 15, 28, 45, ...
    Heptagonal P[7,n]=n(5n−3)/2 --> 1, 7, 18, 34, 55, ...
    Octagonal P[8,n]=n(3n−2)    --> 1, 8, 21, 40, 65, ...

    The ordered set of three 4-digit numbers: 8128, 2882, 8281, has
    three interesting properties.

    The set is cyclic, in that the last two digits of each number is the
    first two digits of the next number (including the last number with
    the first).  Each polygonal type: triangle (P[3,127]=8128), square
    (P[4,91]=8281), and pentagonal (P[5,44]=2882), is represented by a
    different number in the set.  This is the only set of 4-digit numbers
    with this property.  Find the sum of the only ordered set of six
    cyclic 4-digit numbers for which each polygonal type: triangle,
    square, pentagonal, hexagonal, heptagonal, and octagonal, is
    represented by a different number in the set."""
    @memoize
    def is_cyclic(x, y):
        return digits_from_num(x)[2:] == digits_from_num(y)[:2]
    def is_fourdigit(n):
        return number_of_digits(n) == 4
    def make_fourdigit_polygons(*poly_funcs):
        """Return list of lists, each holding only four digit numbers
        of each given polygonal type."""
        # The polygonal type is represented as a function of n, which returns
        # the nth polygonal number. e.g. square(2) == 4, pentagonal(3) == 12
        poly_gen = lambda f: (f(i) for i in range(1, 200)) # 200 is enough
        return [list(filter(is_fourdigit, poly_gen(func))) for func in poly_funcs]
    polys = make_fourdigit_polygons(triangle, square, pentagonal,
                                    hexagonal, heptagonal, octagonal)
    perms = permutations(range(6))
    def check_one(perm):
        return first_true(
            [a, b, c, d, e, f]
            # Use the permutation as indices into 'polys' to check this order
            for a in polys[perm[0]]
            for b in polys[perm[1]] if is_cyclic(a, b)
            for c in polys[perm[2]] if is_cyclic(b, c)
            for d in polys[perm[3]] if is_cyclic(c, d)
            for e in polys[perm[4]] if is_cyclic(d, e)
            for f in polys[perm[5]] if (is_cyclic(e, f) and is_cyclic(f, a)))
    return sum(first_true(check_one(perm) for perm in perms))

def problem62():
    """The cube, 41063625 (345**3), can be permuted to produce two other
    cubes: 56623104 (384**3) and 66430125 (405**3). In fact, 41063625 is
    the smallest cube which has exactly three permutations of its
    digits which are also cube. Find the smallest cube for which
    exactly five permutations of its digits are cube."""
    @tail_recursive
    def process_cube(i=0, cubes=collections.defaultdict(list)):
        # Add i**3 to a dict, the keys of which are an arbitrarily specified
        # canonical permutation of its digits; the values of which are lists of
        # all cube numbers seen so far which are permutable to the key. Every
        # iteration we check whether the list under the current key contains
        # five members, and if so return the smallest member.
        canonical = ''.join(sorted(str(i**3)))
        seen = cubes[canonical]
        seen.append(i**3)
        if len(seen) == 5:
            return min(seen)
        return process_cube(i+1, cubes)
    return process_cube()

def problem63():
    """The 5-digit number, 16807=7**5, is also a fifth power. Similarly,
    the 9-digit number, 134217728=8**9, is a ninth power. How many
    n-digit positive integers exist which are also an nth power?"""
    def find_powers(n):
        """Return the list of powers to which one can raise n such that
        the result of exponentiation is an integer with number_of_digits == power"""
        # find_powers(6) --> [1, 2, 3, 4]
        return takewhile(lambda x: number_of_digits(n**x) == x, count(1))
    # Take results from find_powers(i) for i in count(1) until the empty list
    # indicates no more results. Find the length of all items of every resulting list.
    res = takewhile(lambda x: x != [], ([x for x in find_powers(i)] for i in count(1)))
    return ilen(flatten(res))

def problem64():
    """The first ten continued fraction representations of (irrational) square roots are:

    sqrt(2)=[1;(2)], period=1
    sqrt(3)=[1;(1,2)], period=2
    sqrt(5)=[2;(4)], period=1
    sqrt(6)=[2;(2,4)], period=2
    sqrt(7)=[2;(1,1,1,4)], period=4
    sqrt(8)=[2;(1,4)], period=2
    sqrt(10)=[3;(6)], period=1
    sqrt(11)=[3;(3,6)], period=2
    sqrt(12)= [3;(2,6)], period=2
    sqrt(13)=[3;(1,1,1,1,6)], period=5

    Exactly four continued fractions, for N <= 13, have an odd period. How many
    continued fractions for N <= 10000 have an odd period?"""
    def continued_fraction_sqrt(S):
        # https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Continued_fraction_expansion
        # Using variables S, m, d, a as in the URL above.
        @tail_recursive
        def process_cf(m=0, d=1, a=floor(sqrt(S)), seen=[]):
            seen.append([m, d, a]) # The algortihm terminates when [m, d, a] repeats
            m = (d * a) - m
            d = (S - m**2) / d
            if d == 0: # S is a perfect square.
                return [a]
            a = floor((floor(sqrt(S)) + m) / d)
            if [m, d, a] in seen:
                return [x[2] for x in seen] # The third element is the variable 'a' we want.
            return process_cf(m, d, a, seen)
        return process_cf()
    continued_fractions = (continued_fraction_sqrt(i) for i in range(2, 10000+1))
    odd_period = lambda x: len(x) % 2 == 0 # The first element is not part of the period.
    return quantify(continued_fractions, pred=odd_period)

def problem65():
    """What is most surprising is that the important mathematical constant,
    e = [2; 1,2,1, 1,4,1, 1,6,1 , ... , 1,2k,1, ...].

    The first ten terms in the sequence of convergents for e are:

    2, 3, 8/3, 11/4, 19/7, 87/32, 106/39, 193/71, 1264/465, 1457/536, ...
    The sum of digits in the numerator of the 10th convergent is 1+4+5+7=17.

    Find the sum of digits in the numerator of the 100th convergent of the
    continued fraction for e."""
    def partial_values():
        "Yields the sequence 1,2,1, 1,4,1, 1,6,1, 1,8,1, ..."
        # This is the pattern in the continued fractional representation of e.
        x = 2
        while True:
            yield 1; yield x; yield 1
            x += 2
    def e(n):
        "Returns the nth convergent of the continued fraction for e."
        if n == 1: # The 1st convergent is simply 2.
            return 2
        # Collect the first n-1 partial values of e.
        values = collections.deque(take(n-1, partial_values()))
        # Construct the continued fraction, where 'tail' is the recursive component.
        return Fraction(2 + Fraction(1, tail(values)))
    def tail(values):
        "Recursively returns the tail end of the continued fractional representation of e"
        next = values.popleft()
        if len(values) == 0:
            return next
        return next + Fraction(1, tail(values))
    return sum(digits_from_num(e(100).numerator))

def problem66():
    """Consider quadratic Diophantine equations of the form: x**2 – Dy**2 = 1
    For example, when D=13, the minimal solution in x is 6492 – 13×1802 = 1.

    It can be assumed that there are no solutions in positive integers when D is square.
    By finding minimal solutions in x for D = {2, 3, 5, 6, 7}, we obtain the following:

    3**2 – 2×2**2 = 1
    2**2 – 3×1**2 = 1
    9**2 – 5×4**2 = 1
    5**2 – 6×2**2 = 1
    8**2 – 7×3**2 = 1

    Hence, by considering minimal solutions in x for D ≤ 7, the largest x is obtained when D=5.
    Find the value of D ≤ 1000 in minimal solutions of x for which the largest
    value of x is obtained."""
    def solve_pells_equation(D):
        # Each iteration through the convergents of the continued fraction of
        # sqrt(D), we want to check whether the numerator and denominator
        # provide a solution to the Diophantine equation:
        # https://en.wikipedia.org/wiki/Pell%27s_equation
        # See the section entitled 'Fundamental solution via continued fractions'
        def process_cf(m=0, d=1, a=floor(sqrt(D))):
            # See problem 64 for a link explaining this algorithm. Here we use 'D'
            # in place of 'S' to be consistent with the wording of the question.
            while True:
                yield a
                m = (d * a) - m
                d = (D - m**2) / d
                a = floor((floor(sqrt(D)) + m) / d)
        def convergent(n):
            """Returns the nth convergent of the continued fraction for sqrt(D),
            where D is a non-square positive integer."""
            if n == 1:
                return next(process_cf())
            # Collect the first n partial values of D.
            values = collections.deque(take(n, process_cf()))
            # Construct the continued fraction, where 'tail' is the recursive component.
            return Fraction(values.popleft() + Fraction(1, tail(values)))
        def tail(values):
            "Recursively returns the tail end of the continued fraction for sqrt(D)"
            next = values.popleft()
            if len(values) == 0:
                return next
            return next + Fraction(1, tail(values))
        def is_solution(frac):
            "Check whether the convergent satisfies the Diophantine equation"
            x, y = frac.numerator, frac.denominator
            return x**2 - D*(y**2) == 1
        # Find the solution with the minimal value of x satisfying the equation.
        solution = first_true((convergent(n) for n in count(1)), pred=is_solution)
        # For the purpose of problem 66, we only need the value of x
        return solution.numerator
    solutions = [(i, solve_pells_equation(i)) for i in range(1, 1000+1) if sqrt(i).is_integer() == False]
    # Find the solution wth the largest value of x
    answer = max(solutions, key=lambda s: s[1])
    # Return the value of D for which that value of x was obtained
    return answer[0]

def problem67():
    """Find the maximum total from top to bottom in triangle.txt a 15K text
    file containing a triangle with one-hundred rows."""
    with open("triangle.txt", "r") as f:
        triangle = [list(map(int, row.split())) for row in f]
        @memoize_mutable
        def largest_route(triangle):
            """Recursively find the maximum value of the root node plus the
            largest of its children, and so on, all the way to the base."""
            # where triangle is a list of lists such as [[1], [2, 3], [4, 5, 6]
            # representing a tree of the form:
            #   1
            #  2 3
            # 4 5 6
            root = triangle[0][0]
            if len(triangle) == 1:
                return root
            a, b = child_triangles(triangle)
            return root + max(largest_route(a), largest_route(b))
        def child_triangles(triangle):
            "Split the triangle in two below the root node"
            # [[1], [2, 3], [4, 5, 6]] --> [[2], [4, 5]], [[3], [5, 6]]
            # the two children triangles of the root node.
            a = [row[:-1] for row in triangle[1:]]
            b = [row[1:] for row in triangle[1:]]
            return a, b
        return largest_route(triangle)
