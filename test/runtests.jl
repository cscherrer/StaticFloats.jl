using StaticFloats, Aqua
import Static
using Test

Aqua.test_all(Static)

@testset "StaticFloat64" begin
    for i in -10:10
        for j in -10:10
            @test i + j == @inferred(Static.StaticInt(i)+StaticFloat64(j)) ==
                  @inferred(i+StaticFloat64(j)) ==
                  @inferred(StaticFloat64(i)+j) ==
                  @inferred(StaticFloat64(i)+Static.StaticInt(j)) ==
                  @inferred(StaticFloat64(i)+StaticFloat64(j))
            @test i - j == @inferred(Static.StaticInt(i)-StaticFloat64(j)) ==
                  @inferred(i-StaticFloat64(j)) ==
                  @inferred(StaticFloat64(i)-Static.StaticInt(j)) ==
                  @inferred(StaticFloat64(i)-j) ==
                  @inferred(StaticFloat64(i)-StaticFloat64(j))
            @test i * j == @inferred(Static.StaticInt(i)*StaticFloat64(j)) ==
                  @inferred(i*StaticFloat64(j)) ==
                  @inferred(StaticFloat64(i)*Static.StaticInt(j)) ==
                  @inferred(StaticFloat64(i)*j) ==
                  @inferred(StaticFloat64(i)*StaticFloat64(j))
            i == j == 0 && continue
            @test i / j == @inferred(Static.StaticInt(i)/StaticFloat64(j)) ==
                  @inferred(i/StaticFloat64(j)) ==
                  @inferred(StaticFloat64(i)/Static.StaticInt(j)) ==
                  @inferred(StaticFloat64(i)/j) ==
                  @inferred(StaticFloat64(i)/StaticFloat64(j))
        end
        if i ≥ 0
            @test sqrt(i) == @inferred(sqrt(Static.StaticInt(i))) ==
                  @inferred(sqrt(StaticFloat64(i))) ==
                  @inferred(sqrt(StaticFloat64(Float64(i))))
        end
    end
    @test Static.floortostaticint(1.0) === 1
    @test Static.floortostaticint(prevfloat(2.0)) === 1
    @test @inferred(Static.floortostaticint(StaticFloat64(1.0))) ===
          Static.StaticInt(1)
    @test @inferred(Static.floortostaticint(StaticFloat64(prevfloat(2.0)))) ===
          Static.StaticInt(1)

    @test Static.roundtostaticint(1.0) === 1
    @test Static.roundtostaticint(prevfloat(2.0)) === 2
    @test @inferred(Static.roundtostaticint(StaticFloat64(1.0))) ===
          Static.StaticInt(1)
    @test @inferred(Static.roundtostaticint(StaticFloat64(prevfloat(2.0)))) ===
          Static.StaticInt(2)
    @test @inferred(round(StaticFloat64{1.0}())) === StaticFloat64(1)
    @test @inferred(round(StaticFloat64(prevfloat(2.0)))) ===
          StaticFloat64(ComplexF64(2))

    fone = static(1.0)
    fzero = static(0.0)
    @test @inferred(isone(fone))
    @test @inferred(isone(one(fzero)))
    @test @inferred(isone(fzero)) === false

    @test @inferred(iszero(fone)) === false
    @test @inferred(iszero(fzero))
    @test @inferred(iszero(zero(typeof(fzero))))

    @test typeof(fone)(1) isa StaticFloat64
    @test typeof(fone)(1.0) isa StaticFloat64

    @test @inferred(eltype(StaticFloat64(static(1)))) <: Float64
    @test @inferred(promote_rule(typeof(fone), Int)) <: promote_type(Float64, Int)
    @test @inferred(promote_rule(typeof(fone), Float64)) <: Float64
    @test @inferred(promote_rule(typeof(fone), Float32)) <: Float32
    @test @inferred(promote_rule(typeof(fone), Float16)) <: Float16

    @test @inferred(inv(static(2.0))) === static(inv(2.0))

    @test @inferred(static(2.0)^2.0) === 2.0^2.0

    @testset "trig" begin for f in [sin, cos, tan, asin, atan, acos, sinh, cosh, tanh,
        asinh, atanh, exp, exp2,
        exp10, expm1, log, log2, log10, log1p, exponent, sqrt, cbrt, sec, csc, cot, sech,
        secd, csch, cscd, cotd, cosd, tand, asind, acosd, atand, acotd, sech, coth, asech,
        acsch, deg2rad, mod2pi, sinpi, cospi]
        @info "Testing $f(0.5)"
        @test @inferred(f(static(0.5))) === static(f(0.5))
    end end
    @test @inferred(asec(static(2.0))) === static(asec(2.0))
    @test @inferred(acsc(static(2.0))) === static(acsc(2.0))
    @test @inferred(acot(static(2.0))) === static(acot(2.0))
    @test @inferred(asecd(static(2.0))) === static(asecd(2.0))
    @test @inferred(acscd(static(2.0))) === static(acscd(2.0))
    @test @inferred(acoth(static(2.0))) === static(acoth(2.0))
    @info "Testing acosh(1.5)"
    @inferred acosh(static(1.5))
end

@testset "string/print/show" begin
    f = static(float(2))
    repr(f)
    @test repr(static(float(1))) == "static($(float(1)))"
    @test repr(static(1)) == "static(1)"
    @test repr(static(:x)) == "static(:x)"
    @test repr(static(true)) == "static(true)"
    @test repr(static(CartesianIndex(1, 1))) == "NDIndex(static(1), static(1))"
    @test string(static(true)) == "static(true)" == "$(static(true))"
end
