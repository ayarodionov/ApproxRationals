# ApproxRationals.jl

[![CI](https://github.com/ayarodionov/ApproxRationals/actions/workflows/CI.yml/badge.svg)](https://github.com/ayarodionov/ApproxRationals/actions/workflows/CI.yml)

Rational arithmetic with a **fixed global precision**.

Julia's `Rational` is exact, and that is exactly the problem. Nothing in the
type bounds the size of a numerator or denominator, so in any iterative
computation they grow without limit:

```julia
julia> denominator(sum(1 // big(k) for k in 1:50_000))
# an integer with 21_701 decimal digits
```

Every subsequent operation on that number is slower than the last, and the
extra digits are not information anyone asked for.

`ARational` keeps the rational representation — and therefore exact
comparisons, exact halves and thirds, no binary-fraction surprises — but
rounds after **every** operation to the best rational whose denominator fits
under a global bound:

```julia
using ApproxRationals

setprecision(ARational, 64)          # denominators bounded by 2^64

x = ARational(1, 3) + ARational(1, 7)
# 10//21

denominator(sum(ARational(1, k) for k in 1:50_000))
# a 20-digit integer, and it stays that way forever
```

## How the rounding works

The rounding step is the classical continued-fraction best approximation. Given
the exact result `p/q` and the bound `N`, `bestapprox(p, q, N)` walks the
continued fraction of `p/q`, keeping the convergents

    h_i = a_i·h_{i-1} + h_{i-2},    k_i = a_i·k_{i-1} + k_{i-2}

and stops as soon as `k_i` would exceed `N`. The answer is then either the last
admissible convergent or the best *semiconvergent* below the bound, whichever
is closer — decided by exact integer cross-multiplication, never by floating
point.

This makes the rounding **optimal**, not merely good: no rational with
denominator `≤ N` is strictly closer to the exact result. That is a stronger
guarantee than binary floating point gives you, and it is why the
approximations look like the ones humans use:

```julia
julia> [bestapprox(Rational{BigInt}(BigFloat(π, precision=256)), N) for N in (10, 100, 113, 10^5)]
4-element Vector{Rational{BigInt}}:
      22//7
     311//99
     355//113
  312689//99532
```

`355//113` is accurate to 2.7e-7 with a three-digit denominator; a float with
the same storage cannot do better.

### Relation to the Litvinov–Rodionov–Chourkin scheme

The paper in the [References](#references) parameterises the same idea by
*error* rather than by size: the user gives an absolute tolerance `Δ` and a
relative tolerance `δ`, the continued fraction is truncated at the first
convergent meeting them, and inequality `1/(q_k·(q_k + q_{k+1})) < |p/q −
p_k/q_k| ≤ 1/(q_k·q_{k+1})` lets the error be checked without ever comparing
against the unrounded number. That gives a per-operation error bound directly,
and `Δ = δ = 0` recovers exact arithmetic.

`ApproxRationals` bounds the denominator instead, which is the *fixed-slash*
parameterisation of Matula and Kornerup. The two are duals — a size budget
implies an error bound and vice versa — but they are not the same rounding, and
the difference is exactly why semiconvergents appear here: under an error
budget the first convergent that clears the bar is the right answer, whereas
under a size budget the best convergent is often beaten by a semiconvergent
with a larger, still-admissible denominator (`311//99` beats `22//7` for π at
`N = 100`).

Bounding the size gives predictable memory and a fixed cost per operation;
bounding the error gives a guarantee about the answer. Neither dominates. An
error-controlled mode is not implemented yet.

## Precision control

```julia
setprecision(ARational, 96)              # denominators <= 2^96
setprecision(ARational, 30, base = 10)   # denominators <= 10^30
precision(ARational)                     # bits in the current bound

setmaxdenominator!(1000)                 # bound stated directly
maxdenominator()

with_maxdenominator(10^12) do            # scoped, restored on exit or error
    sum(ARational(1, k) for k in 1:10_000)
end

setprecision(ARational, 128) do
    exp(ARational(1))
end
```

The bound is a plain denominator limit, so precision is continuous — any
integer bound works, not just powers of two.

## What is supported

- `+ - * / ^ inv abs sign` — exact, then rounded.
- `== < <= isless sort hash` — exact on the stored values, no rounding.
- `floor ceil trunc round`, to `ARational` or to any integer type.
- `sqrt cbrt exp log log2 log10 sin cos tan asin acos atan sinh cosh tanh
  atan(y,x)`, and `x^y` for non-integer `y` — evaluated in `BigFloat` at
  generous working precision, then rounded back into the bound.
- Promotion with `Integer`, `Rational` (→ `ARational`) and `AbstractFloat`
  (→ the float type), so `ARational(1,2) + 1//3` and `sort` on mixed arrays
  just work.
- `ARational(π)` and other irrationals.
- `approxerror(x, exact)` to audit how much precision a computation lost.

## Element types

`ARational{BigInt}` is the default and never overflows. When the bound is small
enough, a fixed-width element type avoids `BigInt` entirely:

```julia
setprecision(ARational, 20)
ARational{Int64}(1, 3) + ARational{Int64}(1, 7)     # all machine arithmetic
```

The bound is clamped per element type, so a denominator always fits. Numerators
are converted with a checked conversion and will throw `InexactError` rather
than wrap silently. As a rule of thumb keep `maxdenominator() < 2^31` for
`Int64` and `< 2^63` for `Int128`.

Internally, `ARational{BigInt}` takes a machine fast path: when numerator and
denominator both fit in an `Int128`, the whole reduce-and-round step runs in
`Int128`, which is worth roughly 5–10x at 32–64 bits of precision.

## Performance

Harmonic sum `∑ 1/k`, exact `Rational{BigInt}` against `ARational` at 64 bits
(Apple silicon, Julia 1.12):

| n | exact | denominator | `ARational` | denominator |
|---|---|---|---|---|
| 1 000 | 0.3 ms | 433 digits | 1.8 ms | 20 digits |
| 10 000 | 8.1 ms | 4 345 digits | 17.7 ms | 20 digits |
| 50 000 | 154 ms | 21 701 digits | 92 ms | 20 digits |

Exact arithmetic is quadratic — the operands themselves keep growing — while
bounded arithmetic is linear, so it wins from roughly `n = 15 000` on and the
gap widens without limit after that. Below the crossover the rounding is a real
cost; the point of the bound is predictable memory and predictable time, not
beating exact arithmetic on short computations.

Accuracy on that same sum, against the exact value:

| bound | relative error |
|---|---|
| 2^16 | 5.7e-09 |
| 2^32 | 1.4e-17 |
| 2^64 | 9.5e-37 |
| 2^128 | 4.8e-76 |
| 2^256 | 1.8e-153 |

## Caveats

- **The bound is global mutable state.** That is what makes the arithmetic
  operators work without threading a precision argument through everything, and
  it is the same trade-off `BigFloat` makes. It also means it is not
  thread-safe to change the precision while other threads compute; set it once,
  or use `with_maxdenominator` on a single thread.
- **`==` compares stored values.** Two computations that mathematically agree
  can round differently and compare unequal, exactly as with floats. Compare
  with a tolerance when that matters.
- **There is no smallest positive value other than what the bound allows.**
  Any value below `1/(2·N)` rounds to `0//1` — `ARational(1,3)^100` is `0//1`
  at 40 bits. This is genuine underflow, not a bug: zero really is the closest
  rational under the bound. Unlike floats there is no exponent to fall back on,
  so a computation that ranges over many orders of magnitude wants either a
  large bound or a different representation.
- **Errors accumulate.** Each operation is individually optimal, but the bound
  is not an error bound on a whole computation. Ill-conditioned problems still
  need headroom.

## References

- Grigori Litvinov, Anatoli Rodionov, Andrei Chourkin, *Approximate rational
  arithmetics and arbitrary precision computations* (2001).
  [arXiv:math/0101152](https://arxiv.org/abs/math/0101152) — rational
  arithmetic in which the round-off error, absolute or relative, is controlled
  by the user, with the rounding performed through continued fraction
  expansions. The paper also bounds the work: for an absolute error `10^-N` the
  number of continued-fraction steps is at most `⌊1.672 + 2.392·N⌋`, and on
  average about `N`, so precision costs linear time rather than explosive time.
  See [Relation to the Litvinov–Rodionov–Chourkin
  scheme](#relation-to-the-litvinovrodionovchourkin-scheme) for how this
  package differs.
- D. W. Matula and P. Kornerup, *Finite precision rational arithmetic: slash
  number systems*, IEEE Transactions on Computers **C-34**(1), 3–18 (1985) —
  fixed-slash and floating-slash rationals, the family this package's
  denominator bound belongs to.
- Grigori Litvinov, *Error autocorrection in rational approximation and interval
  estimates (a survey of results)*, Open Mathematics **1**(1), 36–60 (2003).
  [doi:10.2478/BF02475663](https://link.springer.com/article/10.2478/BF02475663)
  — on why errors in rational approximation are often far smaller than a naive
  accumulation argument predicts.

## Running

```
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. examples/demo.jl
```
