# Why bounded-precision rationals: the same computations, exact vs. approximate.
using ApproxRationals, Printf

println("=" ^ 72)
println("1. Harmonic sum  H_n = sum 1/k,  exact Rational{BigInt} vs ARational")
println("=" ^ 72)
setprecision(ARational, 64)
exact_sum(n) = sum(1 // big(k) for k in 1:n)
approx_sum(n) = sum(ARational(1, k) for k in 1:n)
best(f, n) = minimum(@elapsed(f(n)) for _ in 1:3)
exact_sum(20); approx_sum(20)                     # warm up the JIT
for n in (100, 1000, 10_000, 50_000)
    ex, ap = exact_sum(n), approx_sum(n)
    @printf("n=%6d | exact: den %6d digits, %7.1f ms | approx: den %2d digits, %7.1f ms | rel.err %.2e\n",
            n, ndigits(denominator(ex)), 1e3best(exact_sum, n),
            ndigits(denominator(ap)), 1e3best(approx_sum, n),
            Float64(abs(approxerror(ap, ex)) / ex))
end
println("Exact arithmetic grows quadratically (the operands themselves grow);")
println("bounded arithmetic stays linear, so it wins from roughly n = 15_000 on.")

println()
println("=" ^ 72)
println("2. Same computation at several precisions")
println("=" ^ 72)
exact = sum(1 // big(k) for k in 1:2000)
for bits in (16, 32, 64, 128, 256)
    setprecision(ARational, bits)
    ap = sum(ARational(1, k) for k in 1:2000)
    rel = Float64(abs(approxerror(ap, exact)) / exact)
    @printf("bits=%4d  den <= 2^%-4d  value = %-22.16f  rel.err %.3e\n",
            bits, bits, Float64(ap), rel)
end

println()
println("=" ^ 72)
println("3. The rounding is optimal, and it is the approximation humans use")
println("=" ^ 72)
piq = Rational{BigInt}(BigFloat(pi, precision = 256))
for N in (10, 100, 1000, 10^4, 10^5, 10^6)
    r = bestapprox(piq, N)
    @printf("  den <= %-8d ->  %20s   err %.3e   (%d bits of storage)\n",
            N, "$(numerator(r))/$(denominator(r))", abs(Float64(r) - pi),
            ndigits(numerator(r), base = 2) + ndigits(denominator(r), base = 2))
end
println("No rational under each bound is closer -- 355/113 reaches 2.7e-7 with")
println("16 bits total, which a 16-bit float cannot match.")

println()
println("=" ^ 72)
println("4. A feedback loop: x <- 7x/11 + 5/13, 10_000 iterations")
println("=" ^ 72)
setprecision(ARational, 96)
x = ARational(1, 3)
t = @timed for _ in 1:10_000
    global x = x * ARational(7, 11) + ARational(5, 13)
end
fixpoint = (5 // 13) / (1 - 7 // 11)          # = 55//52
@printf("ARational: %8.1f ms, den %d digits, x = %.20f\n",
        1e3t.time, ndigits(denominator(x)), Float64(x))
@printf("fixed point 55//52 = %.20f, error %.3e\n",
        Float64(fixpoint), Float64(abs(approxerror(x, fixpoint))))
println("(exact Rational{BigInt} would need denominators with ~10_000 digits here)")
