using ApproxRationals
using Test
using Random

@testset "ApproxRationals" begin

@testset "bestapprox" begin
    # exact when it fits
    @test bestapprox(3, 4, 100) == (3, 4)
    @test bestapprox(6, 8, 100) == (3, 4)
    @test bestapprox(0, 5, 10) == (0, 1)

    # classic pi approximations
    piq = Rational{BigInt}(BigFloat(π, precision = 256))
    @test bestapprox(piq, 10) == 22 // 7
    @test bestapprox(piq, 100) == 311 // 99      # semiconvergent, better than 22/7
    @test bestapprox(piq, 113) == 355 // 113
    @test bestapprox(piq, 33102) == 103993 // 33102
    # semiconvergents beat 355//113 well before the next convergent shows up
    r = bestapprox(piq, 30000)
    @test denominator(r) <= 30000
    @test abs(r - piq) < abs(355 // 113 - piq)
    for d in 1:30000                     # exhaustive: nothing under the bound is closer
        n = round(BigInt, d * piq)
        @test abs(n // d - piq) >= abs(r - piq)
    end

    # negative values
    @test bestapprox(-piq, 113) == -355 // 113
    @test bestapprox(-22, 7, 3) == (-3, 1)

    # sign normalisation
    n, d = bestapprox(1, -3, 100)
    @test d > 0 && n // d == -1 // 3

    # brute force: no rational with a smaller denominator is closer
    for (p, q) in ((355, 113), (1234567, 7654321), (-98765, 4321)), N in (3, 7, 12, 50)
        n, d = bestapprox(p, q, N)
        @test 0 < d <= N
        err = abs(p * d - n * q) // (q * d)
        for dd in 1:N, nn in (fld(p * dd, q)):(fld(p * dd, q)+1)
            @test abs(p * dd - nn * q) // (q * dd) >= err
        end
    end

    @test_throws ArgumentError bestapprox(1, 3, 0)
    @test_throws DivideError bestapprox(1, 0, 10)
end

@testset "fixed-width bestapprox matches BigInt (no overflow)" begin
    rng = MersenneTwister(20260823)
    for _ in 1:3000
        p = rand(rng, Int64)
        q = rand(rng, 1:typemax(Int64))
        N = rand(rng, 1:typemax(Int64))
        a = bestapprox(Int128(p), Int128(q), Int128(N))
        b = bestapprox(big(p), big(q), big(N))
        @test Int128.(b) == a
        @test 0 < a[2] <= N
    end
    # extremes of the Int128 range still behave
    for (p, q, N) in ((typemax(Int128), typemax(Int128) - 1, big(2)^100),
                      (typemin(Int128) + 1, typemax(Int128), big(2)^64),
                      (typemax(Int128), Int128(3), Int128(7)))
        a = bestapprox(Int128(p), Int128(q), Int128(N))
        @test Int128.(bestapprox(big(p), big(q), big(N))) == a
    end
end

@testset "fast path agrees with the BigInt path" begin
    rng = MersenneTwister(4711)
    for bits in (24, 48, 64, 96, 200)          # 200 bits is past the Int128 gate
        setprecision(ARational, bits)
        bound = maxdenominator()
        for _ in 1:400
            p, q = rand(rng, Int64), rand(rng, 1:typemax(Int64))
            x = ARational(p, q)
            n, d = bestapprox(big(p), big(q), bound)
            @test numerator(x) == n && denominator(x) == d
        end
    end
    setprecision(ARational, 64)
end

@testset "convergents" begin
    c = convergents(355, 113)
    @test c[1] == 3 // 1
    @test c[2] == 22 // 7
    @test last(c) == 355 // 113
    @test issorted(denominator.(c))
end

@testset "construction and conversion" begin
    setprecision(ARational, 64)
    @test Rational(ARational(3, 4)) == 3 // 4
    @test Rational(ARational(6, -8)) == -3 // 4
    @test denominator(ARational(6, -8)) == 4
    @test ARational(5) == ARational(5, 1)
    @test isinteger(ARational(10, 5))
    @test Rational(ARational(0.5)) == 1 // 2
    @test Float64(ARational(1, 4)) === 0.25
    @test float(ARational(1, 8)) === 0.125
    @test ARational(ARational(1, 3)) == ARational(1, 3)
    @test_throws DivideError ARational(1, 0)

    # π rounds to its best approximation under the bound
    setprecision(ARational, 10)
    @test Rational(ARational(π)) == 355 // 113
    setprecision(ARational, 64)
end

@testset "arithmetic is exact when it fits" begin
    setprecision(ARational, 64)
    a, b = ARational(1, 3), ARational(1, 7)
    @test Rational(a + b) == 10 // 21
    @test Rational(a - b) == 4 // 21
    @test Rational(a * b) == 1 // 21
    @test Rational(a / b) == 7 // 3
    @test Rational(-a) == -1 // 3
    @test Rational(inv(a)) == 3 // 1
    @test Rational(a^3) == 1 // 27
    @test Rational(a^-2) == 9 // 1
    @test isone(a^0)
    @test Rational(abs(ARational(-2, 5))) == 2 // 5
    @test iszero(a - a)
end

@testset "denominator never exceeds the bound" begin
    for bits in (8, 16, 24, 53)
        setprecision(ARational, bits)
        bound = maxdenominator()
        x = ARational(1)
        for k in 2:400
            x += ARational(1, k)            # harmonic series: exact denominators explode
            @test denominator(x) <= bound
        end
        y = ARational(1, 3)
        for _ in 1:200
            y = y * ARational(7, 11) + ARational(5, 13)
            @test denominator(y) <= bound
        end
    end
    setprecision(ARational, 64)
end

@testset "rounding is a best approximation" begin
    setprecision(ARational, 12)               # bound 4096
    bound = maxdenominator()
    for (p, q) in ((123456789, 987654321), (1, 100003), (99991, 100003), (-7654321, 1234567))
        x = ARational(p, q)
        @test denominator(x) <= bound
        err = abs(Rational{BigInt}(x) - p // q)
        n, d = bestapprox(big(p), big(q), bound)
        @test err == abs(n // d - p // q)
    end
    setprecision(ARational, 64)
end

@testset "accuracy" begin
    setprecision(ARational, 53)
    # sum of 1/k for k=1..500, compared against the exact rational
    exact = sum(1 // big(k) for k in 1:500)
    approx = sum(ARational(1, k) for k in 1:500)
    @test abs(approxerror(approx, exact)) < 1 // big(10)^12

    # a chain of multiplications tracks the exact value closely
    setprecision(ARational, 80)
    ex = Rational{BigInt}(1)
    ap = ARational(1)
    for k in 1:200
        ex *= (big(k) + 1) // (big(k) + 3)
        ap *= ARational(k + 1, k + 3)
    end
    @test abs(approxerror(ap, ex)) < 1 // big(10)^18
    setprecision(ARational, 64)
end

@testset "underflow: tiny values collapse to zero" begin
    setprecision(ARational, 40)
    @test iszero(ARational(1, 3)^100)          # 3^-100 is far below 1/2^41
    @test iszero(ARational(1, big(2)^60))
    @test !iszero(ARational(1, 3)^20)          # 3^-20 ~ 2.9e-10, still resolvable
    # the boundary: 1/(2N) is where a positive value stops being representable
    bound = maxdenominator()
    @test !iszero(ARational(1, bound))
    @test iszero(ARational(1, 2 * bound + 1))
    setprecision(ARational, 64)
end

@testset "comparison and ordering" begin
    setprecision(ARational, 64)
    @test ARational(1, 3) < ARational(1, 2)
    @test ARational(1, 2) <= ARational(1, 2)
    @test ARational(-1, 2) < ARational(0)
    @test ARational(2, 4) == ARational(1, 2)
    @test !(ARational(1, 2) < ARational(1, 2))
    @test sort([ARational(3, 4), ARational(1, 5), ARational(-2, 3)]) ==
          [ARational(-2, 3), ARational(1, 5), ARational(3, 4)]
    @test maximum([ARational(1, 3), ARational(2, 3)]) == ARational(2, 3)
    @test hash(ARational(1, 2)) == hash(ARational(2, 4))
end

@testset "promotion with Julia numbers" begin
    setprecision(ARational, 64)
    @test ARational(1, 2) + 1 == ARational(3, 2)
    @test 1 + ARational(1, 2) == ARational(3, 2)
    @test 2 * ARational(1, 4) == ARational(1, 2)
    @test ARational(1, 2) + 1 // 3 == ARational(5, 6)
    @test ARational(1, 2) < 1
    @test ARational(1, 2) + 0.25 === 0.75
    @test ARational(1, 2) == 1 // 2
    @test sum(ARational(1, k) for k in 1:4) == ARational(25, 12)
    @test eltype([ARational(1, 2), 1]) == ARational{BigInt}
end

@testset "rounding to integers" begin
    setprecision(ARational, 64)
    @test floor(Int, ARational(7, 2)) == 3
    @test ceil(Int, ARational(7, 2)) == 4
    @test trunc(Int, ARational(-7, 2)) == -3
    @test floor(Int, ARational(-7, 2)) == -4
    @test round(Int, ARational(7, 2)) == 4
    @test round(Int, ARational(5, 2)) == 2       # RoundNearest, ties to even
    @test isinteger(floor(ARational(7, 2)))
end

@testset "elementary functions" begin
    setprecision(ARational, 60)
    @test Float64(sqrt(ARational(2))) ≈ sqrt(2)
    @test Float64(exp(ARational(1))) ≈ ℯ
    @test Float64(log(ARational(ℯ))) ≈ 1 atol = 1e-15
    @test Float64(sin(ARational(1, 2))) ≈ sin(0.5)
    @test Float64(atan(ARational(1), ARational(1))) ≈ pi / 4
    @test Float64(ARational(2)^ARational(1, 2)) ≈ sqrt(2)
    @test ARational(2)^ARational(3) == ARational(8)
    # sqrt(2) is irrational, so the result must be inside the precision bound
    @test denominator(sqrt(ARational(2))) <= maxdenominator()
    setprecision(ARational, 64)
end

@testset "precision control" begin
    old = setprecision(ARational, 20)
    @test maxdenominator() == big(2)^20
    @test precision(ARational) == 20
    setprecision(ARational, 6, base = 10)
    @test maxdenominator() == big(10)^6
    setmaxdenominator!(1000)
    @test maxdenominator() == 1000
    @test Rational(ARational(π)) == 355 // 113

    with_maxdenominator(10) do
        @test Rational(ARational(π)) == 22 // 7
    end
    @test maxdenominator() == 1000

    setprecision(ARational, 30) do
        @test maxdenominator() == big(2)^30
    end
    @test maxdenominator() == 1000

    # restored even when the body throws
    @test_throws ErrorException with_maxdenominator(5) do
        error("boom")
    end
    @test maxdenominator() == 1000

    @test_throws ArgumentError setmaxdenominator!(0)
    @test_throws ArgumentError setprecision(ARational, 0)
    setprecision(ARational, old)
end

@testset "fixed-width element types" begin
    setprecision(ARational, 20)
    x = ARational{Int64}(1, 3)
    y = ARational{Int64}(1, 7)
    @test Rational(x + y) == 10 // 21
    @test x + y isa ARational{Int64}
    z = ARational{Int64}(1)
    for k in 2:200
        z += ARational{Int64}(1, k)
        @test denominator(z) <= maxdenominator()
    end
    # the bound is clamped so a denominator always fits the element type
    setprecision(ARational, 40)
    w = ARational{Int32}(1, 3) + ARational{Int32}(1, 7)
    @test denominator(w) <= typemax(Int32)
    setprecision(ARational, 64)
end

@testset "show" begin
    setprecision(ARational, 64)
    @test sprint(show, ARational(3, 4)) == "3//4"
    @test sprint(show, ARational(-3, 4)) == "-3//4"
    @test occursin("0.75", sprint(show, MIME"text/plain"(), ARational(3, 4)))
    @test sprint(show, MIME"text/plain"(), ARational(4)) == "4//1"
end

@testset "errors" begin
    setprecision(ARational, 64)
    @test_throws DivideError ARational(1, 2) / ARational(0)
    @test_throws DivideError inv(ARational(0))
    @test_throws DivideError ARational(0)^-1
    @test_throws InexactError ARational(Inf)
    @test_throws InexactError ARational(NaN)
end

end
