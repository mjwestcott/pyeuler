#!/usr/bin/python
import string

import data
from toolset import *

def problem1():
    """Add all the natural numbers below 1000 that are multiples of 3 or 5."""
    return sum(x for x in range(1, 1000) if x % 3 == 0 or x % 5 == 0)

def problem2():
    """Find the sum of all the even-valued terms in the Fibonacci < 4 million."""
    even_fibonacci = (x for x in fibonacci() if x % 2)
    return sum(takewhile(lambda x: x < 4e6, even_fibonacci))

def problem3():
    """Find the largest prime factor of a composite number."""
    return max(prime_factors(600851475143))

def problem4():
    """Find the largest palindrome made from the product of two 3-digit numbers."""
    # A brute-force solution is a bit slow, let's try to simplify it a little bit:
    # x*y = "abccda" = 100001a + 10010b + 1100c = 11 * (9091a + 910b + 100c)
    # So at least one of them must be multiple of 11.
    candidates = (x*y for x in range(110, 1000, 11) for y in range(x, 1000))
    return max(x for x in candidates if is_palindromic(x))

def problem5():
    """What is the smallest positive number that is evenly divisible by all of
    the numbers from 1 to 20?."""
    return reduce(least_common_multiple, range(1, 20+1))

def problem6():
    """Find the difference between the sum of the squares of the first one
    hundred natural numbers and the square of the sum."""
    sum_of_squares = sum(x**2 for x in range(1, 100+1))
    square_of_sums = sum(range(1, 100+1))**2
    return square_of_sums - sum_of_squares

def problem7():
    """What is the 10001st prime number?."""
    return nth(get_primes(), 10001-1)

def problem8():
    """Find the greatest product of five consecutive digits in the 1000-digit number"""
    digits = (int(c) for c in "".join(data.problem8.strip().splitlines()))
    return max(product(nums) for nums in groups(digits, 5, 1))

def problem9():
    """There exists exactly one Pythagorean triplet for which a + b + c = 1000.
    Find the product abc."""
    triplets = ((a, b, 1000-a-b) for a in range(1, 999) for b in range(a+1, 999))
    return first(a*b*c for (a, b, c) in triplets if a**2 + b**2 == c**2)

def problem10():
    """Find the sum of all the primes below two million."""
    return sum(takewhile(lambda x: x<2e6, get_primes()))

def problem11():
    """What is the greatest product of four adjacent numbers in any direction
    (up, down, left, right, or diagonally) in the 20x20 grid?"""
    def grid_get(grid, nr, nc, sr, sc):
        """Return cell for coordinate (nr, nc) is a grid of size (sr, sc)."""
        return (grid[nr][nc] if 0 <= nr < sr and 0 <= nc < sc else 0)
    grid = [list(map(int, line.split()))
            for line in data.problem11.strip().splitlines()]
    # For each cell, get 4 groups in directions E, S, SE and SW
    diffs = [(0, +1), (+1, 0), (+1, +1), (+1, -1)]
    sr, sc = len(grid), len(grid[0])
    return max(product(grid_get(grid, nr+i*dr, nc+i*dc, sr, sc) for i in range(4))
               for nr in range(sr)
               for nc in range(sc)
               for (dr, dc) in diffs)

def problem12():
    """What is the value of the first triangle number to have over five
    hundred divisors?"""
    triangle_numbers = (triangle(n) for n in count(1))
    return first(tn for tn in triangle_numbers if ilen(divisors(tn)) > 500)

def problem13():
    """Work out the first ten digits of the sum of the following one-hundred
    50-digit numbers."""
    numbers = (int(x) for x in data.problem13.strip().splitlines())
    return int(str(sum(numbers))[:10])

def problem14():
    """The following iterative sequence is defined for the set of positive
    integers: n -> n/2 (n is even), n -> 3n + 1 (n is odd). Which starting
    number, under one million, produces the longest chain?"""
    def collatz_function(n):
        return ((3*n + 1) if (n % 2) else (n/2))
    @memoize
    def collatz_series_length(n):
        return (1 + collatz_series_length(collatz_function(n)) if n>1 else 0)
    return max(range(1, int(1e6)), key=collatz_series_length)

def problem15():
    """How many routes are there through a 20x20 grid?"""
    # To reach the bottom-right corner in a grid of size n we need to move n times
    # down (D) and n times right (R), in any order. So we can just see the
    # problem as how to put n D's in a 2*n array (a simple permutation),
    # and fill the holes with R's -> permutations(2n, n) = (2n)!/(n!n!) = (2n)!/2n!
    #
    # More generically, this is also a permutation of a multiset
    # which has ntotal!/(n1!*n2!*...*nk!) permutations
    # In this problem the multiset is {n.D, n.R}, so (2n)!/(n!n!) = (2n)!/2n!
    n = 20
    return factorial(2*n) / (factorial(n)**2)

def problem16():
    """What is the sum of the digits of the number 2^1000?"""
    return sum(digits_from_num(2**1000))

def problem17():
    """If all the numbers from 1 to 1000 (one thousand) inclusive were written
    out in words, how many letters would be used?"""
    strings = (get_cardinal_name(n) for n in range(1, 1000+1))
    return ilen(c for c in flatten(strings) if c.isalpha())

def problem18():
    """Find the maximum total from top to bottom of the triangle below:"""
    # The note that go with the problem warns that number 67 presents the same
    # challenge but much bigger, where it won't be possible to solve it using
    # simple brute force. But let's use brute-force here and we'll use the
    # head later. We test all routes from the top of the triangle. We will find
    # out, however, that this brute-force solution is much more complicated to
    # implement (and to understand) than the elegant one.
    def get_numbers(rows):
        """Return groups of "columns" numbers, following all possible ways."""
        for moves in cartesian_product([0, +1], repeat=len(rows)-1):
            indexes = ireduce(operator.add, moves, 0)
            yield (row[index] for (row, index) in zip(rows, indexes))
    rows = [list(map(int, line.split()))
            for line in data.problem18.strip().splitlines()]
    return max(sum(numbers) for numbers in get_numbers(rows))

def problem19():
    """How many Sundays fell on the first of the month during the twentieth
    century (1 Jan 1901 to 31 Dec 2000)?"""
    def is_leap_year(year):
        return (year%4 == 0 and (year%100 != 0 or year%400 == 0))
    def get_days_for_month(year, month):
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return days[month-1] + (1 if (month == 2 and is_leap_year(year)) else 0)
    years_months = ((year, month) for year in range(1901, 2001)
                                  for month in range(1, 12+1))
    # Skip the last month (otherwise we would be checking for 1 Jan 2001)
    days = (get_days_for_month(y, m)
            for (y, m) in years_months
            if (y, m) != (2000, 12))
    # Let's index Monday with 0 and Sunday with 6. 1 Jan 1901 was a Tuesday (1)
    weekday_of_first_day_of_months = ireduce(lambda wd, d: (wd+d) % 7, days, 1)
    return sum(1 for weekday in weekday_of_first_day_of_months if weekday == 6)

def problem20():
    """Find the sum of the digits in the number 100!"""
    return sum(digits_from_num(factorial(100)))

def problem21():
    """Evaluate the sum of all the amicable numbers under 10000."""
    sums = dict((n, sum(proper_divisors(n))) for n in range(1, 10000))
    return sum(a for (a, b) in sums.items() if a != b and sums.get(b, 0) == a)

def problem22():
    """What is the total of all the name scores in the file?"""
    contents = data.openfile("names.txt").read()
    names = sorted(name.strip('"') for name in contents.split(","))
    dictionary = dict((c, n) for (n, c) in enumerate(string.ascii_uppercase, 1))
    return sum(i*sum(dictionary[c] for c in name) for (i, name) in enumerate(names, 1))

def problem23():
    """Find the sum of all the positive integers which cannot be written as
    the sum of two abundant numbers."""
    abundants = set(x for x in range(1, 28123+1) if is_perfect(x) == 1)
    return sum(x for x in range(1, 28123+1)
                 if not any((x-a in abundants) for a in abundants))

def problem24():
    """What is the millionth lexicographic permutation of the digits
    0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?"""
    return num_from_digits(nth(permutations(range(10), 10), int(1e6)-1))

def problem25():
    """What is the first term in the Fibonacci sequence to contain 1000 digits?"""
    # See relation between Fibanacci and the golden-ratio for a non brute-force solution
    return first(idx for (idx, x) in enumerate(fibonacci(), 1) if x >= 10**999)

def problem26():
    """Find the value of d < 1000 for which 1/d contains the longest recurring
    cycle in its decimal fraction part."""
    def division(numerator, denominator):
        """Return (quotient, (decimals, cycle_length)) for numerator / denomominator."""
        def recursive(numerator, denominator, quotients, remainders):
            q, r = divmod(numerator, denominator)
            if r == 0:
                return (quotients + [q], 0)
            elif r in remainders:
                return (quotients, len(remainders) - remainders.index(r))
            else:
                return recursive(10*r, denominator, quotients + [q], remainders + [r])
        decimals = recursive(10*(numerator % denominator), denominator, [], [])
        return (numerator/denominator, decimals)
    # A smarter (and much faster) solution: countdown from 1000 getting cycles'
    # length, and break when a denominator is lower the the current maximum
    # length (since a cycle cannot be larger than the denominator itself).
    return max(range(2, 1000), key=lambda d: division(1, d)[1][1])

def problem27():
    """Find the product of the coefficients, a and b, for the quadratic
    expression that produces the maximum number of primes for consecutive
    values of n, starting with n = 0."""
    def function(n, a, b):
        return n**2 + a*n + b
    def primes_for_a_b(a_b):
        return takewhile(is_prime, (function(n, *a_b) for n in count(0)))
    # b must be prime so n=0 yields a prime (b itself)
    b_candidates = list(x for x in range(1000) if is_prime(x))
    candidates = ((a, b) for a in range(-1000, 1000) for b in b_candidates)
    return product(max(candidates, key=compose(ilen, primes_for_a_b)))

def problem28():
    """What is the sum of the numbers on the diagonals in a 1001 by 1001 spiral
    formed in the same way?"""
    return 1 + sum(4*(n - 2)**2 + 10*(n - 1) for n in range(3, 1001+1, 2))

def problem29():
    """How many distinct terms are in the sequence generated by a**b for
    2 <= a <= 100 and 2 <= b <= 100?"""
    return ilen(unique(a**b for a in range(2, 100+1) for b in range(2, 100+1)))

def problem30():
    """Find the sum of all the numbers that can be written as the sum of fifth
    powers of their digits."""
    candidates = range(2, 6*(9**5))
    return sum(n for n in candidates if sum(x**5 for x in digits_from_num(n)) == n)

def problem31():
    """How many different ways can 2 pounds be made using any number of coins?"""
    def get_weights(units, remaining):
        """Return weights that sum 'remaining'. Pass units in descending order.
        get_weigths([4,2,1], 5) -> (0,0,5), (0,1,3), (0,2,1), (1,0,1)"""
        if len(units) == 1 and remaining % units[0] == 0:
            # Make it generic, do not assume that last unit is 1
            yield (remaining/units[0],)
        elif units:
            for weight in range(0, remaining + 1, units[0]):
                for other_weights in get_weights(units[1:], remaining - weight):
                   yield (weight/units[0],) + other_weights
    coins = [1, 2, 5, 10, 20, 50, 100, 200]
    return ilen(get_weights(sorted(coins, reverse=True), 200))

def problem32():
    """Find the sum of all products whose multiplicand/multiplier/product
    identity can be written as a 1 through 9 pandigital"""
    def get_permutation(ndigits):
        return ((num_from_digits(ds), list(ds))
                for ds in permutations(range(1, 10), ndigits))
    def get_multiplicands(ndigits1, ndigits2):
        return cartesian_product(get_permutation(ndigits1), get_permutation(ndigits2))
    # We have two cases for A * B = C: 'a * bcde = fghi' and 'ab * cde = fghi'
    # Also, since C has always 4 digits, 1e3 <= A*B < 1e4
    candidates = chain(get_multiplicands(1, 4), get_multiplicands(2, 3))
    return sum(unique(a*b for ((a, adigits), (b, bdigits)) in candidates
                          if a*b < 1e4
                          and is_pandigital(adigits + bdigits + digits_from_num(a*b))))

def problem33():
    """There are exactly four non-trivial examples of this type of fraction,
    less than one in value, and containing two digits in the numerator and
    denominator. If the product of these four fractions is given in its lowest
    common terms, find the value of the denominator."""
    def reduce_fraction(num, denom):
        gcd = greatest_common_divisor(num, denom)
        return (num / gcd, denom / gcd)
    def is_curious(numerator, denominator):
        if numerator == denominator or numerator % 10 == 0 or denominator % 10 == 0:
            return False
        # numerator / denominator = ab / cd
        (a, b), (c, d) = map(digits_from_num, [numerator, denominator])
        reduced = reduce_fraction(numerator, denominator)
        return (b == c and reduce_fraction(a, d) == reduced or
                a == d and reduce_fraction(b, c) == reduced)
    curious_fractions = ((num, denom) for num in range(10, 100)
                                      for denom in range(num+1, 100)
                                      if is_curious(num, denom))
    numerator, denominator = map(product, zip(*curious_fractions))
    return reduce_fraction(numerator, denominator)[1]

def problem34():
    """Find the sum of all numbers which are equal to the sum of the factorial
    of their digits."""
    # Cache digits from 0 to 9 to speed it up a little bit
    dfactorials = dict((x, factorial(x)) for x in range(10))

    # Upper bound: ndigits*9! < 10^ndigits -> upper_bound = ndigits*9!
    # That makes 7*9! = 2540160. That's quite a number, so it will be slow.
    #
    # A faster alternative: get combinations with repetition of [0!..9!] in
    # groups of N (1..7), and check the sum value. Note that the upper bound
    # condition is in this case harder to apply.
    upper_bound = first(n*dfactorials[9] for n in count(1) if n*dfactorials[9] < 10**n)
    return sum(x for x in range(3, upper_bound)
                 if x == sum(dfactorials[d] for d in digits_from_num_fast(x)))

def problem35():
    """How many circular primes are there below one million?"""
    def is_circular_prime(digits):
        return all(is_prime(num_from_digits(digits[r:] + digits[:r]))
            for r in range(len(digits)))
    # We will use only 4 digits (1, 3, 7, and 9) to generate candidates, so we
    # must consider the four one-digit primes separately.
    circular_primes = (num_from_digits(ds) for n in range(2, 6+1)
                                           for ds in cartesian_product([1, 3, 7, 9], repeat=n)
                                           if is_circular_prime(ds))
    return ilen(chain([2, 3, 5, 7], circular_primes))

def problem36():
    """Find the sum of all numbers, less than one million, which are
    palindromic in base 10 and base 2."""
    # Apply a basic constraint: a binary number starts with 1, and to be
    # palindromic it must also end with 1, so candidates are odd numbers.
    return sum(x for x in range(1, int(1e6), 2)
                 if is_palindromic(x, base=10)
                 and is_palindromic(x, base=2))

def problem37():
    """Find the sum of the only eleven primes that are both truncatable from
    left to right and right to left."""
    def truncatable_get_primes():
        for ndigits in count(2):
            digit_groups = [[2, 3, 5, 7]] + [[1, 3, 7, 9]]*(ndigits-2) + [[3, 7]]
            for ds in cartesian_product(*digit_groups):
                x = num_from_digits(ds)
                if is_prime(x) and all(is_prime(num_from_digits(ds[n:])) and
                        is_prime(num_from_digits(ds[:-n])) for n in range(1, len(ds))):
                    yield x
    return sum(take(11, truncatable_get_primes()))

def problem38():
    """What is the largest 1 to 9 pandigital 9-digit number that can be formed
    as the concatenated product of an integer with (1,2, ... , n) where n > 1?"""
    def pandigital_concatenated_product(number):
        products = ireduce(operator.add, (digits_from_num(number*x) for x in count(1)))
        candidate_digits = first(ds for ds in products if len(ds) >= 9)
        if len(candidate_digits) == 9 and is_pandigital(candidate_digits):
            return num_from_digits(candidate_digits)
    # 987654321 is the maximum (potential) pandigital,
    # so 9876 is a reasonable upper bound.
    return first(compact(pandigital_concatenated_product(n)
                         for n in range(9876+1, 0, -1)))

def problem39():
    """if p is the perimeter of a right angle triangle with integral length
    sides, {a,b,c}, for which value of p < 1000 is the number of solutions
    maximized?"""
    def get_sides_for_perimeter(perimeter):
        sides = ((perimeter-b-c, b, c) for b in range(1, perimeter//2 + 1)
                                       for c in range(b, perimeter//2 + 1))
        return ((a, b, c) for (a, b, c) in sides if a**2 == b**2 + c**2)
    # Brute-force, check pythagorian triplets for a better solution
    return max(range(120, 1000), key=compose(ilen, get_sides_for_perimeter))

def problem40():
    """An irrational decimal fraction is created by concatenating the positive
    integers: If dn represents the nth digit of the fractional part, find the
    value of the following expression: d1 x d10 x d100 x d1000 x d10000 x
    d100000 x d1000000"""
    def count_digits():
        """Like itertools.count, but returns digits instead. Starts at 1"""
        for nd in count(1):
            for digits in cartesian_product(*([range(1, 10)] + [range(10)]*(nd-1))):
                yield digits
    # We could get a formula for dn, but brute-force is fast enough
    indexes = set([1, 10, 100, 1000, 10000, 100000, 1000000])
    decimals = (d for (idx, d) in enumerate(flatten(count_digits()), 1)
                  if idx in indexes)
    return product(take(len(indexes), decimals))

def problem41():
    """What is the largest n-digit pandigital prime that exists?"""
    # Use the disibility by 3 rule to filter some candidates: if the sum of
    # digits is divisible by 3, so is the number (then it can't be prime).
    maxdigits = first(x for x in range(9, 1, -1) if sum(range(1, x+1)) % 3)
    candidates = (num_from_digits(digits)
                  for ndigits in range(maxdigits, 1, -1)
                  for digits in permutations(range(ndigits, 0, -1), ndigits))
    return first(x for x in candidates if is_prime(x))

def problem42():
    """Using words.txt (right click and 'Save Link/Target As...'), a 16K text
    file containing nearly two-thousand common English words, how many are
    triangle words?"""
    dictionary = dict((c, n) for (n, c) in enumerate(string.ascii_uppercase, 1))
    words = data.openfile("words.txt").read().replace('"', '').split(",")
    return ilen(word for word in words if is_triangle(sum(dictionary[c] for c in word)))

def problem43():
    """The number 1406357289 is a 0 to 9 pandigital number because it is made
    up of each of the digits 0 to 9 in some order, but it also has a rather
    interesting sub-string divisibility property. Let d1 be the 1st digit, d2
    be the 2nd digit, and so on. In this way, we note the following: d2d3d4=406
    is divisible by 2, d3d4d5=063 is divisible by 3, d4d5d6=635 is divisible
    by 5, d5d6d7=357 is divisible by 7, d6d7d8=572 is divisible by 11,
    d7d8d9=728 is divisible by 13, d8d9d10=289 is divisible by 17.
    Find the sum of all 0 to 9 pandigital numbers with this property."""
    # Begin from the last 3-digits and backtrack recursively
    def get_numbers(divisors, candidates, acc_result=()):
        if divisors:
            for candidate in candidates:
                new_acc_result = candidate + acc_result
                if num_from_digits(new_acc_result[:3]) % divisors[0] == 0:
                    new_candidates = [(x,) for x in set(range(10)) - set(new_acc_result)]
                    for res in get_numbers(divisors[1:], new_candidates, new_acc_result):
                        yield res
        else:
            d1 = candidates[0]
            if d1: # d1 is the most significant digit, so it cannot be 0
                yield num_from_digits(d1 + acc_result)
    return sum(get_numbers([17, 13, 11, 7, 5, 3, 2], permutations(range(10), 3)))

def problem44():
    """Find the pair of pentagonal numbers, Pj and Pk, for which their sum
    and difference is pentagonal and D = |Pk - Pj| is minimised; what is the
    value of D?"""
    pairs = ((p1, p2) for (n1, p1) in ((n, pentagonal(n)) for n in count(0))
                      for p2 in (pentagonal(n) for n in range(1, n1))
                      if is_pentagonal(p1-p2) and is_pentagonal(p1+p2))
    p1, p2 = first(pairs)
    return p1 - p2

def problem45():
    """It can be verified that T285 = P165 = H143 = 40755. Find the next
    triangle number that is also pentagonal and hexagonal."""
    # Hexagonal numbers are also triangle, so we'll check
    # only whether they are pentagonal.
    hexagonal_candidates = (hexagonal(x) for x in count(143+1))
    return first(x for x in hexagonal_candidates if is_pentagonal(x))

def problem46():
    """What is the smallest odd composite that cannot be written as the sum
    of a prime and twice a square?"""
    # Primes will be iterated over and over and incremently, so better use a
    # cached generator.
    primes = persistent(get_primes())
    def satisfies_conjecture(x):
        test_primes = takewhile(lambda p: p <  x, primes)
        return any(is_integer(sqrt((x - prime) / 2)) for prime in test_primes)
    odd_composites = (x for x in take_every(2, count(3)) if not is_prime(x))
    return first(x for x in odd_composites if not satisfies_conjecture(x))

def problem47():
    """Find the first four consecutive integers to have four distinct primes
    factors. What is the first of these numbers?"""
    grouped_by_4factors = groupby(count(1), lambda x: len(set(prime_factors(x))) == 4)
    matching_groups = (list(group) for (match, group) in grouped_by_4factors if match)
    return first(grouplst[0] for grouplst in matching_groups if len(grouplst) == 4)

def problem48():
    """Find the last ten digits of the series, 1^1 + 2^2 + 3^3 + ... + 1000^1000"""
    return sum(x**x for x in range(1, 1000+1)) % 10**10

def problem49():
    """The arithmetic sequence, 1487, 4817, 8147, in which each of the terms
    increases by 3330, is unusual in two ways: (i) each of the three terms are
    prime, and, (ii) each of the 4-digit numbers are permutations of one
    another. There are no arithmetic sequences made up of three 1-, 2-, or
    3-digit primes, exhibiting this property, but there is one other 4-digit
    increasing sequence. What 12-digit number do you form by concatenating the
    three terms in this sequence?"""
    def ds(n):
        return set(digits_from_num(n))
    def get_triplets(primes):
        for x1 in sorted(primes):
            for d in range(2, (10000-x1)//2 + 1, 2):
                x2 = x1 + d
                x3 = x1 + 2*d
                if x2 in primes and x3 in primes and ds(x1) == ds(x2) == ds(x3):
                    yield (x1, x2, x3)
    primes = set(takewhile(lambda x: x < 10000, get_primes(1000)))
    solution = nth(get_triplets(primes), 1)
    return num_from_digits(flatten(digits_from_num(x) for x in solution))

def problem50():
    """Which prime, below one-million, can be written as the sum of the most
    consecutive primes?"""
    def get_max_length(primes, n, max_length=0, acc=None):
        if sum(take(max_length, drop(n, primes))) >= 1e6:
            return acc
        accsums = takewhile(lambda acc: acc<1e6, accumulate(drop(n, primes)))
        new_max_length, new_acc = max((idx, acc)
                                      for (idx, acc) in enumerate(accsums)
                                      if is_prime(acc))
        if new_max_length > max_length:
            return get_max_length(primes, n+1, new_max_length, new_acc)
        else:
            return get_max_length(primes, n+1, max_length, acc)
    primes = persistent(get_primes())
    return get_max_length(primes, 0)

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
    multiples = lambda x: (i*x for i in range(2, 7))
    # Check that all digits in the first number are in the second,
    # and that they have the same number of digits.
    same_digits = lambda n, m: (all(d in digits_from_num(m) for d in digits_from_num(n))
                                and number_of_digits(n) == number_of_digits(m))
    return first(n for n in count(1) if all(same_digits(n, m) for m in multiples(n)))

def problem53():
    """How many, not necessarily distinct, values of nCr, for 1 ≤ n ≤ 100, are
    greater than one-million?"""
    C = lambda n, r: factorial(n) / (factorial(r) * factorial(n - r))
    return sum(1 for n in range(1, 101)
                 for r in range(1, n+1)
                 if C(n, r) > 1e6)

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
    group = lambda ranks: sorted(collections.Counter(ranks).values(), reverse=True)
    straightflush = lambda hand: flush(hand) and straight(hand)
    fourofakind = lambda hand: group(ranks(hand)) == [4, 1]
    fullhouse = lambda hand: group(ranks(hand)) == [3, 2]
    flush = lambda hand: len(set(suits(hand))) == 1
    straight = lambda hand: ((max(ranks(hand)) - min(ranks(hand)) == 4)
                             and group(ranks(hand)) == [1, 1, 1, 1, 1])
    threeofakind = lambda hand: group(ranks(hand)) == [3, 1, 1]
    twopair = lambda hand: group(ranks(hand)) == [2, 2, 1]
    onepair = lambda hand: group(ranks(hand)) == [2, 1, 1, 1]
    # Return a value for the hand and its ranks to break ties
    value = lambda hand: ((8 if straightflush(hand) else
                           7 if fourofakind(hand) else
                           6 if fullhouse(hand) else
                           5 if flush(hand) else
                           4 if straight(hand) else
                           3 if threeofakind(hand) else
                           2 if twopair(hand) else
                           1 if onepair(hand) else
                           0), ranks(hand))
    compare = lambda hand1, hand2: 1 if max((hand1, hand2), key=value) == hand1 else 0
    players = lambda row: (row[:5], row[5:])
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
    return max(digit_sum(a**b) for a in range(100)
                               for b in range(100))

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
    # The repeating pattern at the end of the expansions
    tail = lambda x: 2 + Fraction(1, x)
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
        if primes/total < 0.10:
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
