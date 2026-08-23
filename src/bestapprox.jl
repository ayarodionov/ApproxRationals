# Best rational approximation with a bounded denominator, via continued fractions.

"""
    bestapprox(p::Integer, q::Integer, N::Integer) -> (n, d)

Best rational approximation `n/d` of `p/q` subject to `0 < d <= N`.

"Best" is meant in the strong sense: no rational with denominator `<= N` is
strictly closer to `p/q`. The result is reduced and has `d > 0`.

The algorithm walks the continued fraction expansion of `p/q`, keeping the
convergents `h_i/k_i` produced by

    h_i = a_i*h_{i-1} + h_{i-2},   k_i = a_i*k_{i-1} + k_{i-2}

As soon as `k_i > N` the search stops and the answer is either the last
admissible convergent `h_{i-1}/k_{i-1}` or the best *semiconvergent*
`(a'*h_{i-1} + h_{i-2}) / (a'*k_{i-1} + k_{i-2})` with the largest `a' <= a_i`
that still fits under `N`. The two candidates are compared exactly (integer
cross-multiplication), so the choice never depends on floating point.

Every intermediate stays bounded by `max(|p|, q)`, so the routine never
overflows a fixed-width element type that could hold the inputs, and costs
`O(log N)` iterations of small-integer arithmetic.
"""
function bestapprox(p::T, q::T, N::T) where {T<:Integer}
    q == 0 && throw(DivideError())
    N < 1 && throw(ArgumentError("denominator bound must be >= 1, got $N"))
    if q < 0
        p, q = -p, -q
    end

    # convergent recurrences, seeded with h_{-1}/k_{-1} = 1/0 and h_{-2}/k_{-2} = 0/1
    hm1, hm2 = one(T), zero(T)
    km1, km2 = zero(T), one(T)

    x, y = p, q
    while y != 0
        a = fld(x, y)
        x, y = y, x - a * y          # x - a*y == mod(x, y), so y stays >= 0

        # Would the next convergent denominator a*km1 + km2 break the bound?
        # Asked this way instead of forming the product, so that no intermediate
        # can overflow a fixed-width element type: every value in this loop stays
        # bounded by max(|p|, q).
        if !iszero(km1) && a > fld(N - km2, km1)
            # Largest semiconvergent index still under the bound.
            ap = fld(N - km2, km1)
            hs = ap * hm1 + hm2
            ks = ap * km1 + km2
            # |p/q - hs/ks| vs |p/q - hm1/km1|, cleared of denominators.
            # Widened: this is the only place a triple product appears.
            W = widen(T)
            lhs = abs(W(p) * W(ks) - W(hs) * W(q)) * W(km1)
            rhs = abs(W(p) * W(km1) - W(hm1) * W(q)) * W(ks)
            return lhs < rhs ? (hs, ks) : (hm1, km1)
        end

        h = a * hm1 + hm2
        k = a * km1 + km2

        hm2, hm1 = hm1, h
        km2, km1 = km1, k
    end
    return (hm1, km1)               # p/q itself was representable
end

bestapprox(p::Integer, q::Integer, N::Integer) = bestapprox(promote(p, q, N)...)

"""
    bestapprox(r::Rational, N::Integer) -> Rational

Best approximation of `r` with denominator at most `N`.
"""
function bestapprox(r::Rational{T}, N::Integer) where {T<:Integer}
    n, d = bestapprox(numerator(r), denominator(r), T(N))
    return n // d
end

"""
    convergents(p::Integer, q::Integer) -> Vector{Rational}

All continued-fraction convergents of `p/q`, in order of increasing
denominator. Useful for inspecting how fast an approximation is converging.
"""
function convergents(p::Integer, q::Integer)
    p, q = promote(p, q)
    q == 0 && throw(DivideError())
    if q < 0
        p, q = -p, -q
    end
    T = typeof(p)
    out = Rational{T}[]
    hm1, hm2 = one(T), zero(T)
    km1, km2 = zero(T), one(T)
    x, y = p, q
    while y != 0
        a = fld(x, y)
        x, y = y, x - a * y
        hm2, hm1 = hm1, a * hm1 + hm2
        km2, km1 = km1, a * km1 + km2
        push!(out, hm1 // km1)
    end
    return out
end

convergents(r::Rational) = convergents(numerator(r), denominator(r))
