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
       bestapprox, approxwithin, convergents,
       RoundingScheme, SizeBound, ErrorBound,
       roundingscheme, setrounding!, with_rounding,
       maxdenominator, setmaxdenominator!, with_maxdenominator,
       approxerror, isexact

include("bestapprox.jl")

# ---------------------------------------------------------------------------
# global precision
# ---------------------------------------------------------------------------

const DEFAULT_BITS = 64

"""
    RoundingScheme

How a rational is rounded after every operation. Two schemes are available,
and they are duals of each other:

- [`SizeBound`](@ref) caps the denominator. Memory and per-operation cost are
  then fixed and known in advance; the error follows from the cap.
- [`ErrorBound`](@ref) caps the error. The accuracy of every single operation
  is then guaranteed; the size follows from the tolerance.

Neither dominates. Pick the one whose guarantee you actually need.
"""
abstract type RoundingScheme end

"""
    SizeBound(maxden::Integer)

Round to the best rational with denominator at most `maxden`, using
[`bestapprox`](@ref). This is the *fixed-slash* scheme of Matula and Kornerup,
refined to admit semiconvergents so the result is a true best approximation.
"""
struct SizeBound <: RoundingScheme
    maxden::BigInt
    function SizeBound(maxden::Integer)
        maxden < 1 && throw(ArgumentError("denominator bound must be >= 1, got $maxden"))
        return new(big(maxden))
    end
end

"""
    ErrorBound(; abstol = nothing, reltol = nothing, threshold = 0)

Round to the first continued-fraction convergent meeting the given tolerances,
using [`approxwithin`](@ref) — the scheme of Litvinov, Rodionov and Chourkin.

- `abstol` (the paper's `Δ`) bounds `|rounded − exact|`.
- `reltol` (the paper's `δ`) bounds `|rounded − exact| / |exact|`.
- `nothing` means an infinite tolerance, i.e. that criterion is not applied;
  at least one of the two must be given. Both criteria must hold when both are.
- A tolerance of `0` means exact: that operand is never rounded.
- `threshold` (the paper's `M`) leaves a result alone entirely while its
  numerator and denominator both have at most that many decimal digits, so
  short exact fractions survive a coarse tolerance.

Tolerances are converted to exact rationals, so `ErrorBound(abstol = 1e-8)`
uses the exact binary value of that float. Pass a `Rational` for a decimal
tolerance: `ErrorBound(abstol = 1//10^8)`.

Unlike `SizeBound` this puts no cap on the denominator, but it does not need
one: an absolute tolerance `Δ` holds denominators to roughly `1/sqrt(Δ)`.
"""
struct ErrorBound <: RoundingScheme
    abstol::Union{Nothing,Rational{BigInt}}
    reltol::Union{Nothing,Rational{BigInt}}
    threshold::Int
    function ErrorBound(; abstol = nothing, reltol = nothing, threshold::Integer = 0)
        at = _tolerance(abstol, "abstol")
        rt = _tolerance(reltol, "reltol")
        at === nothing && rt === nothing &&
            throw(ArgumentError("give abstol, reltol, or both; with neither, nothing is rounded"))
        threshold < 0 && throw(ArgumentError("threshold must be >= 0, got $threshold"))
        return new(at, rt, Int(threshold))
    end
end

_tolerance(::Nothing, ::AbstractString) = nothing
function _tolerance(x::Real, name::AbstractString)
    x < 0 && throw(ArgumentError("$name must be >= 0, got $x"))
    isinf(x) && return nothing
    isnan(x) && throw(ArgumentError("$name must not be NaN"))
    return Rational{BigInt}(x)
end

const SCHEME = Ref{RoundingScheme}(SizeBound(big(2)^DEFAULT_BITS))

"""
    roundingscheme() -> RoundingScheme

The scheme currently applied after every operation.
"""
roundingscheme() = SCHEME[]

"""
    setrounding!(s::RoundingScheme) -> RoundingScheme

Install `s` as the global rounding scheme and return the previous one, so it
can be restored with another `setrounding!`.
"""
function setrounding!(s::RoundingScheme)
    old = SCHEME[]
    SCHEME[] = s
    return old
end

"""
    with_rounding(f, s::RoundingScheme)

Run `f()` under scheme `s`, restoring the previous scheme afterwards (also on
error).

```julia
with_rounding(ErrorBound(abstol = 1//10^20)) do
    sum(ARational(1, k) for k in 1:1000)
end
```
"""
function with_rounding(f, s::RoundingScheme)
    old = setrounding!(s)
    try
        return f()
    finally
        SCHEME[] = old
    end
end

# --- size-bound conveniences ----------------------------------------------

"""
    maxdenominator() -> BigInt

Current global denominator bound. Throws if the active scheme bounds error
rather than size — see [`roundingscheme`](@ref).
"""
maxdenominator() = _maxden(SCHEME[])
_maxden(s::SizeBound) = s.maxden
_maxden(::ErrorBound) = throw(ArgumentError(
    "the active rounding scheme bounds error, not denominator size; see roundingscheme()"))

"""
    setmaxdenominator!(n::Integer) -> RoundingScheme

Switch to `SizeBound(n)` and return the previous scheme.
"""
setmaxdenominator!(n::Integer) = setrounding!(SizeBound(n))

"""
    with_maxdenominator(f, n::Integer)

Run `f()` under `SizeBound(n)`, restoring the previous scheme afterwards.

```julia
with_maxdenominator(10^12) do
    sum(ARational(1, k) for k in 1:1000)
end
```
"""
with_maxdenominator(f, n::Integer) = with_rounding(f, SizeBound(n))

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

# Hard cap the element type imposes. Both parts must be storable in T, so it is
# not enough to hold the denominator under typemax(T): a value of magnitude v
# has a numerator about v times its denominator, and that has to fit too.
_typecap(::Type{BigInt}, ::Integer, ::Integer) = nothing
function _typecap(::Type{T}, n::Integer, d::Integer) where {T<:Integer}
    m = big(typemax(T))
    scale = cld(abs(big(n)), big(d)) + 1        # an upper bound on |n/d| + 1
    return max(big(1), fld(m, scale))
end

# --- applying a scheme to one reduced fraction ------------------------------

function _round(s::SizeBound, ::Type{T}, n::W, d::W) where {T<:Integer,W<:Integer}
    cap = _typecap(T, n, d)
    lim = cap === nothing ? s.maxden : min(s.maxden, cap)
    return d > lim ? bestapprox(n, d, W(lim)) : (n, d)
end

function _round(s::ErrorBound, ::Type{T}, n::W, d::W) where {T<:Integer,W<:Integer}
    cap = _typecap(T, n, d)
    if !(s.threshold > 0 && ndigits(n) <= s.threshold && ndigits(d) <= s.threshold)
        B = _errbound(s, n, d)
        if B !== nothing
            if W === BigInt || B <= big(typemax(W))
                n, d = approxwithin(n, d, W(B))
            else
                # The tolerance is finer than this element type can express;
                # give the closest thing it can hold instead of overflowing.
                n, d = bestapprox(n, d, W(cap))
            end
        end
    end
    # The error criterion caps no denominator, so a fixed-width element type
    # still needs its own net.
    if cap !== nothing && d > W(cap)
        n, d = bestapprox(n, d, W(cap))
    end
    return (n, d)
end

"""
    _errbound(s::ErrorBound, n, d) -> BigInt or nothing

The `B` to hand [`approxwithin`](@ref) so that `n/d` is rounded within every
tolerance in `s`; `nothing` when `n/d` must be kept exactly.

`|err| <= 1/B` is the guarantee, so an absolute tolerance `Δ` needs
`B >= 1/Δ`, and a relative tolerance `δ` needs `B >= q/(δ·|p|)`. Requiring both
means taking the larger.
"""
function _errbound(s::ErrorBound, n::Integer, d::Integer)
    B = big(1)
    if s.abstol !== nothing
        iszero(s.abstol) && return nothing
        B = max(B, cld(denominator(s.abstol), numerator(s.abstol)))
    end
    if s.reltol !== nothing
        iszero(s.reltol) && return nothing
        iszero(n) && return nothing                 # zero is exact; no relative error
        B = max(B, cld(big(d) * denominator(s.reltol),
                       numerator(s.reltol) * abs(big(n))))
    end
    return B
end

# Can this scheme do its work for n/d inside an Int128?
_fastok(s::SizeBound, ::BigInt, ::BigInt) = s.maxden <= _FAST_LIMIT
function _fastok(s::ErrorBound, n::BigInt, d::BigInt)
    B = _errbound(s, n, d)
    return B === nothing || B <= _FAST_LIMIT
end

"""
    _make(T, n, d)

Reduce `n/d`, round it under the active scheme, and wrap it as `ARational{T}`.
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
    n, d = _round(SCHEME[], T, n, d)
    return ARational{T}(T(n), T(d), Val(:raw))
end

# Fast path for the BigInt default: whenever numerator and denominator both fit
# in an Int128, the reduce-and-round step runs entirely in machine arithmetic.
# The continued fraction is O(log N) divisions long, and running those on BigInt
# costs far more than the allocations it saves — this is worth roughly a 5-10x
# speedup at 32-64 bits of precision. Both kernels keep every intermediate under
# max(|p|, q) and widen their few triple products, so Int128 inputs are safe.
const _FAST_LIMIT = big(typemax(Int128))

_fits_fast(x::BigInt) = -_FAST_LIMIT <= x <= _FAST_LIMIT

function _make(::Type{BigInt}, n::BigInt, d::BigInt)
    iszero(d) && throw(DivideError())
    if d < 0
        n, d = -n, -d
    end
    s = SCHEME[]
    if _fits_fast(n) && d <= _FAST_LIMIT && _fastok(s, n, d)
        p, q = Int128(n), Int128(d)
        g = gcd(p, q)
        if !isone(g)
            p = div(p, g)
            q = div(q, g)
        end
        p, q = _round(s, BigInt, p, q)
        return ARational{BigInt}(BigInt(p), BigInt(q), Val(:raw))
    end
    g = gcd(n, d)
    if !isone(g)
        n = div(n, g)
        d = div(d, g)
    end
    n, d = _round(s, BigInt, n, d)
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

# Via Rational rather than F(num)/F(den): the parts can be far outside the
# float's range even when their quotient sits comfortably inside it.
(::Type{F})(x::ARational) where {F<:AbstractFloat} = F(x.num // x.den)
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
Base.precision(::Type{<:ARational}) = ndigits(maxdenominator(), base = 2) - 1
Base.precision(::ARational) = Base.precision(ARational)

"""
    setprecision(ARational, bits::Integer) -> Int
    setprecision(f, ARational, bits::Integer)

Set the denominator bound to `2^bits` (`base = 10` gives `10^digits`).
The second form restores the old precision when `f` returns.
"""
function Base.setprecision(::Type{<:ARational}, bits::Integer; base::Integer = 2)
    bits < 1 && throw(ArgumentError("precision must be >= 1 bit, got $bits"))
    return setrounding!(SizeBound(big(base)^bits))
end

function Base.setprecision(f::Function, ::Type{A}, bits::Integer; base::Integer = 2) where {A<:ARational}
    bits < 1 && throw(ArgumentError("precision must be >= 1 bit, got $bits"))
    return with_rounding(f, SizeBound(big(base)^bits))
end

# working precision for the BigFloat-backed elementary functions: enough
# headroom that the final rounding, not the float evaluation, is the error.
_workbits() = _workbits(SCHEME[])
_workbits(s::SizeBound) = 2 * ndigits(s.maxden, base = 2) + 32
function _workbits(s::ErrorBound)
    B = big(1)
    for tol in (s.abstol, s.reltol)
        tol !== nothing && !iszero(tol) &&
            (B = max(B, cld(denominator(tol), numerator(tol))))
    end
    return 2 * ndigits(B, base = 2) + 64
end

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
