"""
    ApproxRationals

Rational arithmetic with a *fixed global precision*.

Exact rational arithmetic is mathematically perfect but practically unusable
for long computations: numerators and denominators grow without bound, and
after a few thousand operations every `+` costs more than the whole rest of
the program. This package keeps rationals honest by rounding after **every**
operation: the exact result is computed, then replaced by the best rational
approximation whose denominator does not exceed a global bound.

The rounding step is the classical continued-fraction best-approximation
(see [`bestapprox`](@ref)), so each operation is optimal in the strongest
sense available: no other rational under the bound is closer to the exact
answer.

```julia
setprecision(ARational, 32)          # denominators bounded by 2^32
x = ARational(1, 3) + ARational(1, 7)
```
"""
module ApproxRationals

export ARational,
       bestapprox, convergents,
       maxdenominator, setmaxdenominator!, with_maxdenominator,
       approxerror, isexact

include("bestapprox.jl")

# ---------------------------------------------------------------------------
# global precision
# ---------------------------------------------------------------------------

const DEFAULT_BITS = 64
const MAXDEN = Ref{BigInt}(big(2)^DEFAULT_BITS)

"""
    maxdenominator() -> BigInt

Current global denominator bound. Every `ARational` produced by arithmetic has
a denominator no larger than this.
"""
maxdenominator() = MAXDEN[]

"""
    setmaxdenominator!(n::Integer) -> BigInt

Set the global denominator bound to `n` and return the previous value.
"""
function setmaxdenominator!(n::Integer)
    n < 1 && throw(ArgumentError("denominator bound must be >= 1, got $n"))
    old = MAXDEN[]
    MAXDEN[] = big(n)
    return old
end

"""
    with_maxdenominator(f, n::Integer)

Run `f()` with the denominator bound temporarily set to `n`, restoring the
previous bound afterwards (also on error).

```julia
with_maxdenominator(10^12) do
    sum(ARational(1, k) for k in 1:1000)
end
```
"""
function with_maxdenominator(f, n::Integer)
    old = setmaxdenominator!(n)
    try
        return f()
    finally
        MAXDEN[] = old
    end
end

# ---------------------------------------------------------------------------
# the type
# ---------------------------------------------------------------------------

"""
    ARational{T<:Integer} <: Real

A rational number stored as a reduced `num/den` pair with `den > 0` and
`den <= maxdenominator()`. Arithmetic is exact-then-rounded: operands are
combined exactly and the result is passed through [`bestapprox`](@ref).

`ARational(n, d)`, `ARational(x)` for integers, rationals and floats. The
element type defaults to `BigInt`, which never overflows; `ARational{Int64}`
and friends are available when the denominator bound is small enough to make
that safe (roughly `maxdenominator() < 2^31` for `Int64`).
"""
struct ARational{T<:Integer} <: Real
    num::T
    den::T
    # unchecked: caller guarantees reduced, den > 0, den <= bound
    ARational{T}(n::T, d::T, ::Val{:raw}) where {T<:Integer} = new{T}(n, d)
end

# Denominator bound usable for element type T (a value of T can never exceed
# typemax(T), so clamp there for fixed-width integers).
_denlimit(::Type{BigInt}) = MAXDEN[]
_denlimit(::Type{T}) where {T<:Integer} = min(MAXDEN[], big(typemax(T)))

"""
    _make(T, n, d)

Reduce `n/d`, round it to the global precision, and wrap it as `ARational{T}`.
`n` and `d` may be of a wider type than `T`.
"""
function _make(::Type{T}, n::W, d::W) where {T<:Integer,W<:Integer}
    iszero(d) && throw(DivideError())
    if d < 0
        n, d = -n, -d
    end
    g = gcd(n, d)
    if !isone(g)
        n = div(n, g)
        d = div(d, g)
    end
    lim = W(_denlimit(T))
    if d > lim
        n, d = bestapprox(n, d, lim)
    end
    return ARational{T}(T(n), T(d), Val(:raw))
end

# Fast path for the BigInt default: whenever numerator and denominator both fit
# in an Int128, the reduce-and-round step runs entirely in machine arithmetic.
# The continued fraction is O(log N) divisions long, and running those on BigInt
# costs far more than the allocations it saves — this is worth roughly a 5-10x
# speedup at 32-64 bits of precision. `bestapprox` keeps every intermediate
# under max(|p|, q) and widens its one triple product, so Int128 inputs are safe.
const _FAST_LIMIT = big(typemax(Int128))

_fits_fast(x::BigInt) = -_FAST_LIMIT <= x <= _FAST_LIMIT

function _make(::Type{BigInt}, n::BigInt, d::BigInt)
    iszero(d) && throw(DivideError())
    if d < 0
        n, d = -n, -d
    end
    lim = MAXDEN[]
    if lim <= _FAST_LIMIT && _fits_fast(n) && d <= _FAST_LIMIT
        p, q = Int128(n), Int128(d)
        g = gcd(p, q)
        if !isone(g)
            p = div(p, g)
            q = div(q, g)
        end
        if q > Int128(lim)
            p, q = bestapprox(p, q, Int128(lim))
        end
        return ARational{BigInt}(BigInt(p), BigInt(q), Val(:raw))
    end
    g = gcd(n, d)
    if !isone(g)
        n = div(n, g)
        d = div(d, g)
    end
    if d > lim
        n, d = bestapprox(n, d, lim)
    end
    return ARational{BigInt}(n, d, Val(:raw))
end

_make(::Type{T}, n::Integer, d::Integer) where {T<:Integer} = _make(T, promote(n, d)...)

# public constructors ------------------------------------------------------

ARational{T}(n::Integer, d::Integer) where {T<:Integer} = _make(T, widen(T)(n), widen(T)(d))
ARational{T}(n::Integer) where {T<:Integer} = ARational{T}(n, one(T))
ARational{T}(r::Rational) where {T<:Integer} = ARational{T}(numerator(r), denominator(r))
ARational{T}(x::ARational) where {T<:Integer} = ARational{T}(x.num, x.den)
ARational{T}(x::AbstractFloat) where {T<:Integer} =
    (isfinite(x) || throw(InexactError(:ARational, ARational{T}, x));
     ARational{T}(Rational{BigInt}(x)))
ARational{T}(x::AbstractIrrational) where {T<:Integer} =
    ARational{T}(_rat_from_big(BigFloat(x, precision = _workbits())))

ARational(n::Integer, d::Integer) = ARational{BigInt}(n, d)
ARational(x::Union{Integer,Rational,AbstractFloat,AbstractIrrational}) = ARational{BigInt}(x)
ARational(x::ARational) = x

# BigFloat -> exact Rational{BigInt} (works for any finite BigFloat)
function _rat_from_big(x::BigFloat)
    isfinite(x) || throw(InexactError(:ARational, ARational{BigInt}, x))
    iszero(x) && return 0 // 1
    m, e = frexp(x)                       # x == m * 2^e, 0.5 <= |m| < 1
    p = precision(x)
    num = BigInt(ldexp(m, p))             # exact integer
    ex = e - p
    return ex >= 0 ? (num * big(2)^ex) // 1 : num // (big(2)^(-ex))
end

# ---------------------------------------------------------------------------
# basic accessors
# ---------------------------------------------------------------------------

Base.numerator(x::ARational) = x.num
Base.denominator(x::ARational) = x.den

Base.Rational{T}(x::ARational) where {T<:Integer} = T(x.num) // T(x.den)
Base.Rational(x::ARational{T}) where {T} = x.num // x.den

Base.zero(::Type{ARational{T}}) where {T} = ARational{T}(zero(T), one(T), Val(:raw))
Base.one(::Type{ARational{T}}) where {T} = ARational{T}(one(T), one(T), Val(:raw))
Base.zero(x::ARational) = zero(typeof(x))
Base.one(x::ARational) = one(typeof(x))

Base.iszero(x::ARational) = iszero(x.num)
Base.isone(x::ARational) = isone(x.num) && isone(x.den)
Base.isinteger(x::ARational) = isone(x.den)
Base.isfinite(::ARational) = true
Base.isnan(::ARational) = false
Base.isinf(::ARational) = false
Base.sign(x::ARational{T}) where {T} = ARational{T}(T(sign(x.num)), one(T), Val(:raw))
Base.signbit(x::ARational) = signbit(x.num)
Base.abs(x::ARational{T}) where {T} = ARational{T}(abs(x.num), x.den, Val(:raw))
Base.:-(x::ARational{T}) where {T} = ARational{T}(-x.num, x.den, Val(:raw))
Base.inv(x::ARational{T}) where {T} =
    iszero(x.num) ? throw(DivideError()) :
    (x.num > 0 ? ARational{T}(x.den, x.num, Val(:raw)) :
                 ARational{T}(-x.den, -x.num, Val(:raw)))

"""
    isexact(x::ARational, r::Rational) -> Bool

Whether `x` represents `r` with no rounding at all.
"""
isexact(x::ARational, r::Rational) = x.num * denominator(r) == numerator(r) * x.den

# ---------------------------------------------------------------------------
# arithmetic: exact, then rounded to the global precision
# ---------------------------------------------------------------------------

for (op, expr) in ((:+, :(xn * yd + yn * xd)),
                   (:-, :(xn * yd - yn * xd)))
    @eval function Base.$op(x::ARational{T}, y::ARational{T}) where {T}
        W = widen(T)
        xn, xd = W(x.num), W(x.den)
        yn, yd = W(y.num), W(y.den)
        return _make(T, $expr, xd * yd)
    end
end

function Base.:*(x::ARational{T}, y::ARational{T}) where {T}
    W = widen(T)
    # cross-reduce first: keeps the exact product small before rounding
    a = gcd(W(x.num), W(y.den))
    b = gcd(W(y.num), W(x.den))
    return _make(T, div(W(x.num), a) * div(W(y.num), b),
                    div(W(x.den), b) * div(W(y.den), a))
end

function Base.:/(x::ARational{T}, y::ARational{T}) where {T}
    iszero(y.num) && throw(DivideError())
    W = widen(T)
    a = gcd(W(x.num), W(y.num))
    b = gcd(W(x.den), W(y.den))
    return _make(T, div(W(x.num), a) * div(W(y.den), b),
                    div(W(x.den), b) * div(W(y.num), a))
end

Base.://(x::ARational, y::ARational) = x / y

function Base.:^(x::ARational{T}, n::Integer) where {T}
    n == 0 && return one(ARational{T})
    if n < 0
        iszero(x.num) && throw(DivideError())
        return inv(x)^(-n)
    end
    # binary powering with rounding at every step, so nothing ever blows up
    result = one(ARational{T})
    base = x
    while n > 0
        if isodd(n)
            result = result * base
        end
        n >>= 1
        n > 0 && (base = base * base)
    end
    return result
end

# ---------------------------------------------------------------------------
# comparison (exact on the stored values)
# ---------------------------------------------------------------------------

function Base.:(==)(x::ARational{T}, y::ARational{T}) where {T}
    return x.num == y.num && x.den == y.den   # both reduced with den > 0
end

function Base.isless(x::ARational{T}, y::ARational{T}) where {T}
    W = widen(T)
    return W(x.num) * W(y.den) < W(y.num) * W(x.den)
end

Base.:<(x::ARational, y::ARational) = isless(x, y)
Base.:<=(x::ARational, y::ARational) = !isless(y, x)
Base.hash(x::ARational, h::UInt) = hash(x.num // x.den, h)
Base.cmp(x::ARational, y::ARational) = isless(x, y) ? -1 : (x == y ? 0 : 1)

# ---------------------------------------------------------------------------
# promotion / conversion
# ---------------------------------------------------------------------------

Base.promote_rule(::Type{ARational{T}}, ::Type{ARational{S}}) where {T,S} =
    ARational{promote_type(T, S)}
Base.promote_rule(::Type{ARational{T}}, ::Type{S}) where {T,S<:Integer} =
    ARational{promote_type(T, S)}
Base.promote_rule(::Type{ARational{T}}, ::Type{Rational{S}}) where {T,S} =
    ARational{promote_type(T, S)}
Base.promote_rule(::Type{ARational{T}}, ::Type{S}) where {T,S<:AbstractFloat} = S

Base.convert(::Type{ARational{T}}, x::ARational{T}) where {T} = x
Base.convert(::Type{ARational{T}}, x::Union{Integer,Rational,AbstractFloat,ARational}) where {T} =
    ARational{T}(x)

(::Type{F})(x::ARational) where {F<:AbstractFloat} = F(x.num) / F(x.den)
Base.float(x::ARational) = Float64(x)
Base.AbstractFloat(x::ARational) = Float64(x)
Base.big(x::ARational) = BigFloat(x.num, precision = _workbits()) / BigFloat(x.den, precision = _workbits())

Base.trunc(::Type{T}, x::ARational) where {T<:Integer} = T(div(x.num, x.den))
Base.floor(::Type{T}, x::ARational) where {T<:Integer} = T(fld(x.num, x.den))
Base.ceil(::Type{T}, x::ARational) where {T<:Integer} = T(cld(x.num, x.den))
Base.round(::Type{T}, x::ARational) where {T<:Integer} = T(round(Rational(x), RoundNearest))
for f in (:trunc, :floor, :ceil, :round)
    @eval Base.$f(x::ARational{T}) where {T} = ARational{T}($f(BigInt, x), one(BigInt))
end

# ---------------------------------------------------------------------------
# precision, Julia-style
# ---------------------------------------------------------------------------

"""
    precision(ARational) -> Int

Number of bits in the current denominator bound, i.e. `floor(log2(bound))`.
"""
Base.precision(::Type{<:ARational}) = ndigits(MAXDEN[], base = 2) - 1
Base.precision(::ARational) = Base.precision(ARational)

"""
    setprecision(ARational, bits::Integer) -> Int
    setprecision(f, ARational, bits::Integer)

Set the denominator bound to `2^bits` (`base = 10` gives `10^digits`).
The second form restores the old precision when `f` returns.
"""
function Base.setprecision(::Type{<:ARational}, bits::Integer; base::Integer = 2)
    bits < 1 && throw(ArgumentError("precision must be >= 1 bit, got $bits"))
    old = Base.precision(ARational)
    setmaxdenominator!(big(base)^bits)
    return old
end

function Base.setprecision(f::Function, ::Type{A}, bits::Integer; base::Integer = 2) where {A<:ARational}
    old = MAXDEN[]
    try
        setprecision(A, bits; base = base)
        return f()
    finally
        MAXDEN[] = old
    end
end

# working precision for the BigFloat-backed elementary functions: enough
# headroom that the final rounding, not the float evaluation, is the error.
_workbits() = 2 * ndigits(MAXDEN[], base = 2) + 32

# ---------------------------------------------------------------------------
# elementary functions: evaluated in BigFloat, rounded back
# ---------------------------------------------------------------------------

for f in (:sqrt, :cbrt, :exp, :expm1, :log, :log2, :log10, :log1p,
          :sin, :cos, :tan, :asin, :acos, :atan,
          :sinh, :cosh, :tanh, :asinh, :acosh, :atanh)
    @eval function Base.$f(x::ARational{T}) where {T}
        setprecision(BigFloat, _workbits()) do
            ARational{T}(_rat_from_big($f(big(x))))
        end
    end
end

function Base.:^(x::ARational{T}, y::ARational{T}) where {T}
    isinteger(y) && return x^numerator(y)
    setprecision(BigFloat, _workbits()) do
        ARational{T}(_rat_from_big(big(x)^big(y)))
    end
end

function Base.atan(y::ARational{T}, x::ARational{T}) where {T}
    setprecision(BigFloat, _workbits()) do
        ARational{T}(_rat_from_big(atan(big(y), big(x))))
    end
end

# ---------------------------------------------------------------------------
# error reporting
# ---------------------------------------------------------------------------

"""
    approxerror(x::ARational, exact::Union{Rational,Real}) -> Rational{BigInt}

Signed error `x - exact` as an exact rational (when `exact` is rational), or
as a `Float64` otherwise. Handy for auditing how much precision a computation
actually lost.
"""
approxerror(x::ARational, exact::Rational) = Rational{BigInt}(x) - Rational{BigInt}(exact)
approxerror(x::ARational, exact::Integer) = Rational{BigInt}(x) - big(exact)
approxerror(x::ARational, exact::ARational) = Rational{BigInt}(x) - Rational{BigInt}(exact)
approxerror(x::ARational, exact::Real) = Float64(x) - Float64(exact)

# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------

Base.show(io::IO, x::ARational) = print(io, x.num, "//", x.den)

function Base.show(io::IO, ::MIME"text/plain", x::ARational{T}) where {T}
    print(io, x.num, "//", x.den)
    if !isone(x.den)
        print(io, "  (", Float64(x), ")")
    end
end

end # module ApproxRationals
