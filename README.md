# Introduction

http://projecteuler.net/

Project Euler is a website dedicated to a series of computational problems intended to be solved with computer programs. The project attracts adults and students interested in mathematics and computer programming. Here we will solve them using Python with purely functional programming style.

Some people have expressed their concerns about making the solutions of the Euler Project public. While the risk of having people using them unjustly certainly exists (shame on them), the benefits of sharing knowledge should exceed this petty inconvenient.

# On functional programming

If "functional programming" sounds unfamiliar to you, read these pages and discover this exciting paradigm:

- http://en.wikipedia.org/wiki/Functional_programming
- http://linuxgazette.net/109/pramode.html
- http://www.ibm.com/developerworks/library/l-prog.html
- http://www.amk.ca/python/writing/functional

Python is not a purely functional language, so some (many, I am afraid) of these solutions won't be as pythonic as one would like. But this is the challenge: solve the Euler Project problems using pre FP, state variables are absolutely forbidden!

If you are a Python novice this is probably not the best place to start. But if you already are familiar with Python and want to explore its functional possibilities, it may be interesting.

# Get the solutions

To check the correctness of the solutions we can download a text file from this project (not related to pyeuler):

    $ wget http://projecteuler-solutions.googlecode.com/svn/trunk/Solutions.txt

# Run the problems

    $ python pyeuler/run.py --solutions-file Solutions.txt
    1: 233168 (ok)
    2: 4613732 (ok)
    ...

    $ python pyeuler/run.py --solutions-file Solutions.txt {1..3} 5
    1: 233168 (ok)
    2: 4613732 (ok)
    3: 6857 (ok)
    5: 232792560 (ok)
