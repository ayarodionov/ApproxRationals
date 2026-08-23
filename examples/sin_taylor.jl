# Computing sin(x) from its Taylor series.
#
#     sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
#
# This converges for every x, but for x away from zero it converges *through*
# enormous intermediate terms. At x = 50 the largest term is about 2.9e20, and
# they all have to cancel down to an answer smaller than 1 — some 68 bits of
# cancellation. Float64 has no chance: its error is proportional to the size of
# the numbers it is holding, so by the time the terms have cancelled, the error
# they carried is larger than the answer.
#
# Run with:  julia --project=. examples/sin_taylor.jl

using ApproxRationals, Printf, Random

"""
    taylorsin(x; atol) -> typeof(x)

Sum the Taylor series for `sin(x)` until a term falls below `atol`. Written
generically, so the identical code runs in `Float64`, in exact
`Rational{BigInt}`, and in `ARational` under either rounding scheme.
"""
function taylorsin(x::T; atol = 1e-25) where {T}
    total = zero(x)
    term = x
    k = 1
    while abs(float(term)) >= atol && k < 400
        total += term
        k += 2
        term = -term * x * x / T((k - 1) * k)
    end
    return total
end

println("=" ^ 76)
println("sin(x) by Taylor series")
println("=" ^ 76)
@printf("%5s  %-26s %-26s %s\n", "x", "Float64", "ARational, 2^53 bound", "exact")
println("-" ^ 76)

for x in (1, 5, 15, 30, 50)
    ref = sin(Float64(x))
    f64 = taylorsin(Float64(x))
    art = setprecision(ARational, 53) do
        Float64(taylorsin(ARational(x)))
    end
    ex = Float64(taylorsin(Rational{BigInt}(x)))
    @printf("%5d  %-26.15f %-26.15f %.15f\n", x, f64, art, ex)
    @printf("%5s  err %-22.1e err %-22.1e err %.1e\n",
            "", abs(f64 - ref), abs(art - ref), abs(ex - ref))
end

println()
println("Float64 loses the answer entirely at x = 50 — off by 4e4 — while a rational")
println("with denominators capped at 2^53 gets every digit. Both carry \"53 bits\";")
println("the difference is what those bits are measured against.")

println()
println("=" ^ 76)
println("Why: a size bound is an *absolute* error bound")
println("=" ^ 76)
println("Rounding p/q to denominator <= N leaves an error of about 1/(q·N), which")
println("does not depend on how large p/q is. A float's error is proportional to the")
println("value. Cancellation destroys relative accuracy and leaves absolute accuracy")
println("untouched, which is exactly the asymmetry the series above exploits.")
println()
setprecision(ARational, 53)
rng = MersenneTwister(7)
println("  worst absolute error over 2000 roundings, denominators capped at 2^53:")
for e in (0, 5, 10, 20, 40)
    worst = 0.0
    for _ in 1:2000
        v = (big(10)^e * rand(rng, big(10)^19:big(10)^20)) // rand(rng, big(10)^19:big(10)^20)
        worst = max(worst, Float64(abs(approxerror(ARational(numerator(v), denominator(v)), v))))
    end
    @printf("    values of magnitude ~1e%-2d   %.1e\n", e, worst)
end
@printf("  Float64 at magnitude 1e20 cannot do better than %.1e.\n", eps(1e20))

println()
println("=" ^ 76)
println("Saying it with an error bound instead")
println("=" ^ 76)
println("ErrorBound(abstol=...) asks for the same guarantee directly, and needs no")
println("guess about how much cancellation is coming. ErrorBound(reltol=...) asks for")
println("the float-like guarantee — and fails the same way Float64 does. This is")
println("variant IV of Table 1 in the paper.")
println()

for x in (15, 30, 50)
    ref = sin(Float64(x))
    for (label, scheme) in ("abstol=1e-20" => ErrorBound(abstol = big(10)^-20),
                            "reltol=1e-20" => ErrorBound(reltol = big(10)^-20))
        r, den = with_rounding(scheme) do
            v = taylorsin(ARational(x))
            (Float64(v), ndigits(denominator(v)))
        end
        @printf("  x = %2d  %s  ->  %-22.15f err %-9.1e (denominator: %2d digits)\n",
                x, label, r, abs(r - ref), den)
    end
end

println()
println("=" ^ 76)
println("Cost")
println("=" ^ 76)
exact(x) = taylorsin(Rational{BigInt}(x))
approx(x) = setprecision(ARational, 53) do
    taylorsin(ARational(x))
end
exact(5); approx(5)                                    # warm up the JIT
best(f, x) = minimum(@elapsed(f(x)) for _ in 1:5)

@printf("%5s  %-32s %s\n", "x", "exact Rational{BigInt}", "ARational, 2^53 bound")
for x in (15, 30, 50)
    e, a = exact(x), approx(x)
    @printf("%5d  %4d-digit den, %6.2f ms       %4d-digit den, %6.2f ms\n",
            x, ndigits(denominator(e)), 1e3best(exact, x),
            ndigits(denominator(a)), 1e3best(approx, x))
end
println()
println("This series is only ~100 terms, so exact arithmetic never gets big enough to")
println("hurt and is the faster choice here. What the bound buys is a denominator that")
println("stays put: it is the same size at x = 50 as at x = 15, where the exact one has")
println("quintupled. That is what matters when the computation does not stop at 100")
println("operations — see section 1 of demo.jl for where the time crossover lands.")
