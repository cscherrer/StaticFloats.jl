module StaticFloats

export StaticFloat64, NDIndex
export dynamic, is_static, known, static, static_promote

using Static: StaticInt
import Static

abstract type StaticFloat{N} <: Real end

function Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, ::StaticFloat{N}) where {N}
    checkindex(Bool, inds, N)
end

"""
    StaticFloat64{N}

A statically sized `Float64`.
Use `static(N)` instead of `Val(N)` when you want it to behave like a number.
"""
struct StaticFloat64{N} <: StaticFloat{N}
    StaticFloat64{N}() where {N} = new{N::Float64}()
    StaticFloat64(x::Float64) = new{x}()
    StaticFloat64(x::Int) = new{Base.sitofp(Float64, x)::Float64}()
    StaticFloat64(x::StaticInt{N}) where {N} = StaticFloat64(convert(Float64, N))
    StaticFloat64(x::Complex) = StaticFloat64(convert(Float64, x))
end

const FloatOne = StaticFloat64{one(Float64)}
const FloatZero = StaticFloat64{zero(Float64)}

Base.eltype(@nospecialize(T::Type{<:StaticFloat64})) = Float64


_fill_to_length(x::Tuple{Vararg{Any, N}}, n::StaticInt{N}) where {N} = x
@inline function _fill_to_length(x::Tuple{Vararg{Any, M}}, n::StaticInt{N}) where {M, N}
    return _fill_to_length((x..., static(1)), n)
end

_flatten(i::StaticInt{N}) where {N} = (i,)
_flatten(i::Integer) = (Int(i),)
_flatten(i::Base.AbstractCartesianIndex) = _flatten(Tuple(i)...)
@inline _flatten(i::StaticInt, I...) = (i, _flatten(I...)...)
@inline _flatten(i::Integer, I...) = (Int(i), _flatten(I...)...)
@inline function _flatten(i::Base.AbstractCartesianIndex, I...)
    return (_flatten(Tuple(i)...)..., _flatten(I...)...)
end

"""
    known(::Type{T})

Returns the known value corresponding to a static type `T`. If `T` is not a static type then
`nothing` is returned.

See also: [`static`](@ref), [`is_static`](@ref)
"""
known(::Type{<:StaticFloat{T}}) where {T} = T
known(::Type{Val{V}}) where {V} = V
_get_known(::Type{T}, dim::StaticInt{D}) where {T, D} = known(field_type(T, dim))
known(@nospecialize(T::Type{<:Tuple})) = eachop(_get_known, nstatic(Val(fieldcount(T))), T)
known(T::DataType) = nothing
known(@nospecialize(x)) = known(typeof(x))

"""
    static(x)

Returns a static form of `x`. If `x` is already in a static form then `x` is returned. If
there is no static alternative for `x` then an error is thrown.

See also: [`is_static`](@ref), [`known`](@ref)

```julia
julia> using Static

julia> static(1)
static(1)

julia> static(true)
True()

julia> static(:x)
static(:x)

```
"""
static(@nospecialize(x::StaticFloat)) = x
static(x::Integer) = StaticInt(x)
function static(x::Union{AbstractFloat, Complex, Rational, AbstractIrrational})
    StaticFloat64(Float64(x))
end
static(x) = Static.static(x)
static(x::Tuple{Vararg{Any}}) = map(static, x)
static(::Val{V}) where {V} = static(V)

Static.is_static(@nospecialize(x::Type{<:StaticFloat})) = Static.True()

@inline Static.dynamic(@nospecialize x::StaticFloat) = known(x)


StaticType{X} = Union{StaticFloat{X}, Static.StaticType{X}}

"""
    static_promote(x, y)

Throws an error if `x` and `y` are not equal, preferentially returning the one that is known
at compile time.
"""
@inline static_promote(x::StaticType{X}, ::StaticType{X}) where {X} = x
@noinline function static_promote(::StaticType{X}, ::StaticType{Y}) where {X, Y}
    error("$X and $Y are not equal")
end
Base.@propagate_inbounds function static_promote(::StaticType{N}, x) where {N}
    static(static_promote(N, x))
end
Base.@propagate_inbounds function static_promote(x, ::StaticType{N}) where {N}
    static(static_promote(N, x))
end
Base.@propagate_inbounds static_promote(x, y) = _static_promote(x, y)
Base.@propagate_inbounds function _static_promote(x, y)
    @boundscheck x === y || error("$x and $y are not equal")
    x
end
_static_promote(::Nothing, ::Nothing) = nothing
_static_promote(x, ::Nothing) = x
_static_promote(::Nothing, y) = y

Base.@propagate_inbounds function _promote_shape(a::Tuple{A, Vararg{Any}},
                                                 b::Tuple{B, Vararg{Any}}) where {A, B}
    (static_promote(getfield(a, 1), getfield(b, 1)),
     _promote_shape(Base.tail(a), Base.tail(b))...)
end
_promote_shape(::Tuple{}, ::Tuple{}) = ()
Base.@propagate_inbounds function _promote_shape(::Tuple{}, b::Tuple{B}) where {B}
    (static_promote(static(1), getfield(b, 1)),)
end
Base.@propagate_inbounds function _promote_shape(a::Tuple{A}, ::Tuple{}) where {A}
    (static_promote(static(1), getfield(a, 1)),)
end

function Base.promote_rule(@nospecialize(T1::Type{<:StaticFloat}),
                           @nospecialize(T2::Type{<:StaticFloat}))
    promote_rule(eltype(T1), eltype(T2))
end
function Base.promote_rule(::Type{<:Base.TwicePrecision{R}},
                           @nospecialize(T::Type{<:StaticFloat})) where {R <: Number}
    promote_rule(Base.TwicePrecision{R}, eltype(T))
end
function Base.promote_rule(@nospecialize(T1::Type{<:StaticFloat}),
                           T2::Type{<:Union{Rational, AbstractFloat, Signed}})
    promote_rule(T2, eltype(T1))
end

Base.:(~)(::StaticFloat{N}) where {N} = static(~N)

Base.inv(x::StaticFloat{N}) where {N} = one(x) / x

@inline Base.one(@nospecialize T::Type{<:StaticFloat}) = static(one(eltype(T)))
@inline Base.zero(@nospecialize T::Type{<:StaticFloat}) = static(zero(eltype(T)))
@inline Base.iszero(::StaticFloat64{0.0}) = true
@inline Base.iszero(@nospecialize x::StaticFloat) = false
@inline Base.isone(::Union{FloatOne}) = true
@inline Base.isone(@nospecialize x::StaticFloat) = false
@inline Base.iseven(@nospecialize x::StaticFloat) = iseven(known(x))
@inline Base.isodd(@nospecialize x::StaticFloat) = isodd(known(x))

Base.AbstractFloat(x::StaticFloat) = StaticFloat64(x)

Base.abs(::StaticFloat{N}) where {N} = static(abs(N))
Base.abs2(::StaticFloat{N}) where {N} = static(abs2(N))
Base.sign(::StaticFloat{N}) where {N} = static(sign(N))

Base.widen(@nospecialize(x::StaticFloat)) = widen(known(x))

function Base.convert(::Type{T}, @nospecialize(N::StaticFloat)) where {T <: Number}
    convert(T, known(N))
end

#Base.Bool(::StaticInt{N}) where {N} = Bool(N)

(::Type{T})(x::StaticFloat) where {T <: Real} = T(known(x))
function (@nospecialize(T::Type{<:StaticFloat}))(x::Union{AbstractFloat,
                                                           AbstractIrrational, Integer,
                                                           Rational})
    static(convert(eltype(T), x))
end

@inline Base.:(-)(::StaticFloat{N}) where {N} = static(-N)
Base.:(*)(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = static(X * Y)
Base.:(/)(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = static(X / Y)
Base.:(-)(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = static(X - Y)
Base.:(+)(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = static(X + Y)

@generated Base.sqrt(::StaticFloat{N}) where {N} = :($(static(sqrt(N))))

function Base.div(::StaticFloat{X}, ::StaticFloat{Y}, m::RoundingMode) where {X, Y}
    static(div(X, Y, m))
end
Base.div(x::Real, ::StaticFloat{Y}, m::RoundingMode) where {Y} = div(x, Y, m)
Base.div(::StaticFloat{X}, y::Real, m::RoundingMode) where {X} = div(X, y, m)

Base.rem(@nospecialize(x::StaticFloat), T::Type{<:Integer}) = rem(known(x), T)
Base.rem(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = static(rem(X, Y))
Base.rem(x::Real, ::StaticFloat{Y}) where {Y} = rem(x, Y)
Base.rem(::StaticFloat{X}, y::Real) where {X} = rem(X, y)

Base.mod(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = static(mod(X, Y))

Base.round(::StaticFloat64{M}) where {M} = StaticFloat64(round(M))
roundtostaticint(::StaticFloat64{M}) where {M} = StaticInt(round(Int, M))
roundtostaticint(x::AbstractFloat) = round(Int, x)
floortostaticint(::StaticFloat64{M}) where {M} = StaticInt(Base.fptosi(Int, M))
floortostaticint(x::AbstractFloat) = Base.fptosi(Int, x)

Base.:(==)(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = ==(X, Y)

Base.:(<)(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = <(X, Y)

Base.isless(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = isless(X, Y)
Base.isless(::StaticFloat{X}, y::Real) where {X} = isless(X, y)

Base.min(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = static(min(X, Y))
Base.min(::StaticFloat{X}, y::Number) where {X} = min(X, y)
Base.min(x::Number, ::StaticFloat{Y}) where {Y} = min(x, Y)

Base.max(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = static(max(X, Y))
Base.max(::StaticFloat{X}, y::Number) where {X} = max(X, y)
Base.max(x::Number, ::StaticFloat{Y}) where {Y} = max(x, Y)

Base.minmax(::StaticFloat{X}, ::StaticFloat{Y}) where {X, Y} = static(minmax(X, Y))

Base.real(@nospecialize(x::StaticFloat)) = x
Base.real(@nospecialize(T::Type{<:StaticFloat})) = eltype(T)
Base.imag(@nospecialize(x::StaticFloat)) = zero(x)

Base.rad2deg(::StaticFloat64{M}) where {M} = StaticFloat64(rad2deg(M))
Base.deg2rad(::StaticFloat64{M}) where {M} = StaticFloat64(deg2rad(M))
@generated Base.cbrt(::StaticFloat64{M}) where {M} = StaticFloat64(cbrt(M))
Base.mod2pi(::StaticFloat64{M}) where {M} = StaticFloat64(mod2pi(M))
@generated Base.sinpi(::StaticFloat64{M}) where {M} = StaticFloat64(sinpi(M))
@generated Base.cospi(::StaticFloat64{M}) where {M} = StaticFloat64(cospi(M))
Base.exp(::StaticFloat64{M}) where {M} = StaticFloat64(exp(M))
Base.exp2(::StaticFloat64{M}) where {M} = StaticFloat64(exp2(M))
Base.exp10(::StaticFloat64{M}) where {M} = StaticFloat64(exp10(M))
@generated Base.expm1(::StaticFloat64{M}) where {M} = StaticFloat64(expm1(M))
@generated Base.log(::StaticFloat64{M}) where {M} = StaticFloat64(log(M))
@generated Base.log2(::StaticFloat64{M}) where {M} = StaticFloat64(log2(M))
@generated Base.log10(::StaticFloat64{M}) where {M} = StaticFloat64(log10(M))
@generated Base.log1p(::StaticFloat64{M}) where {M} = StaticFloat64(log1p(M))
@generated Base.sin(::StaticFloat64{M}) where {M} = StaticFloat64(sin(M))
@generated Base.cos(::StaticFloat64{M}) where {M} = StaticFloat64(cos(M))
@generated Base.tan(::StaticFloat64{M}) where {M} = StaticFloat64(tan(M))
Base.sec(x::StaticFloat64{M}) where {M} = inv(cos(x))
Base.csc(x::StaticFloat64{M}) where {M} = inv(sin(x))
Base.cot(x::StaticFloat64{M}) where {M} = inv(tan(x))
@generated Base.asin(::StaticFloat64{M}) where {M} = StaticFloat64(asin(M))
@generated Base.acos(::StaticFloat64{M}) where {M} = StaticFloat64(acos(M))
@generated Base.atan(::StaticFloat64{M}) where {M} = StaticFloat64(atan(M))
@generated Base.sind(::StaticFloat64{M}) where {M} = StaticFloat64(sind(M))
@generated Base.cosd(::StaticFloat64{M}) where {M} = StaticFloat64(cosd(M))
Base.tand(x::StaticFloat64{M}) where {M} = sind(x) / cosd(x)
Base.secd(x::StaticFloat64{M}) where {M} = inv(cosd(x))
Base.cscd(x::StaticFloat64{M}) where {M} = inv(sind(x))
Base.cotd(x::StaticFloat64{M}) where {M} = inv(tand(x))
Base.asind(x::StaticFloat64{M}) where {M} = rad2deg(asin(x))
Base.acosd(x::StaticFloat64{M}) where {M} = rad2deg(acos(x))
Base.asecd(x::StaticFloat64{M}) where {M} = rad2deg(asec(x))
Base.acscd(x::StaticFloat64{M}) where {M} = rad2deg(acsc(x))
Base.acotd(x::StaticFloat64{M}) where {M} = rad2deg(acot(x))
Base.atand(x::StaticFloat64{M}) where {M} = rad2deg(atan(x))
@generated Base.sinh(::StaticFloat64{M}) where {M} = StaticFloat64(sinh(M))
Base.cosh(::StaticFloat64{M}) where {M} = StaticFloat64(cosh(M))
Base.tanh(x::StaticFloat64{M}) where {M} = StaticFloat64(tanh(M))
Base.sech(x::StaticFloat64{M}) where {M} = inv(cosh(x))
Base.csch(x::StaticFloat64{M}) where {M} = inv(sinh(x))
Base.coth(x::StaticFloat64{M}) where {M} = inv(tanh(x))
@generated Base.asinh(::StaticFloat64{M}) where {M} = StaticFloat64(asinh(M))
@generated Base.acosh(::StaticFloat64{M}) where {M} = StaticFloat64(acosh(M))
@generated Base.atanh(::StaticFloat64{M}) where {M} = StaticFloat64(atanh(M))
Base.asech(x::StaticFloat64{M}) where {M} = acosh(inv(x))
Base.acsch(x::StaticFloat64{M}) where {M} = asinh(inv(x))
Base.acoth(x::StaticFloat64{M}) where {M} = atanh(inv(x))
Base.asec(x::StaticFloat64{M}) where {M} = acos(inv(x))
Base.acsc(x::StaticFloat64{M}) where {M} = asin(inv(x))
Base.acot(x::StaticFloat64{M}) where {M} = atan(inv(x))

@inline Base.exponent(::StaticFloat{M}) where {M} = static(exponent(M))

Base.:(^)(::StaticFloat64{x}, y::Float64) where {x} = exp2(log2(x) * y)

@inline function invariant_permutation(@nospecialize(x::Tuple), @nospecialize(y::Tuple))
    if y === x === ntuple(static, StaticInt(nfields(x)))
        return True()
    else
        return False()
    end
end

@inline nstatic(::Val{N}) where {N} = ntuple(StaticInt, Val(N))

permute(@nospecialize(x::Tuple), @nospecialize(perm::Val)) = permute(x, static(perm))
@inline function permute(@nospecialize(x::Tuple), @nospecialize(perm::Tuple))
    if invariant_permutation(nstatic(Val(nfields(x))), perm) === False()
        return eachop(getindex, perm, x)
    else
        return x
    end
end

"""
    eachop(op, args...; iterator::Tuple{Vararg{StaticInt}}) -> Tuple

Produces a tuple of `(op(args..., iterator[1]), op(args..., iterator[2]),...)`.
"""
@inline function eachop(op::F, itr::Tuple{T, Vararg{Any}}, args::Vararg{Any}) where {F, T}
    (op(args..., first(itr)), eachop(op, Base.tail(itr), args...)...)
end
eachop(::F, ::Tuple{}, args::Vararg{Any}) where {F} = ()

"""
    eachop_tuple(op, arg, args...; iterator::Tuple{Vararg{StaticInt}}) -> Type{Tuple}

Produces a tuple type of `Tuple{op(arg, args..., iterator[1]), op(arg, args..., iterator[2]),...}`.
Note that if one of the arguments passed to `op` is a `Tuple` type then it should be the first argument
instead of one of the trailing arguments, ensuring type inference of each element of the tuple.
"""
eachop_tuple(op, itr, arg, args...) = _eachop_tuple(op, itr, arg, args)
@generated function _eachop_tuple(op, ::I, arg, args::A) where {A, I}
    t = Expr(:curly, Tuple)
    narg = length(A.parameters)
    for p in I.parameters
        call_expr = Expr(:call, :op, :arg)
        if narg > 0
            for i in 1:narg
                push!(call_expr.args, :(getfield(args, $i)))
            end
        end
        push!(call_expr.args, :(StaticInt{$(p.parameters[1])}()))
        push!(t.args, call_expr)
    end
    Expr(:block, Expr(:meta, :inline), t)
end

#=
    find_first_eq(x, collection::Tuple)

Finds the position in the tuple `collection` that is exactly equal (i.e. `===`) to `x`.
If `x` and `collection` are static (`is_static`) and `x` is in `collection` then the return
value is a `StaticInt`.
=#
@generated function find_first_eq(x::X, itr::I) where {X, N, I <: Tuple{Vararg{Any, N}}}
    # we avoid incidental code gen when evaluated a tuple of known values by iterating
    #  through `I.parameters` instead of `known(I)`.
    index = known(X) === nothing ? nothing : findfirst(==(X), I.parameters)
    if index === nothing
        :(Base.Cartesian.@nif $(N + 1) d->(dynamic(x) == dynamic(getfield(itr, d))) d->(d) d->(nothing))
    else
        :($(static(index)))
    end
end

# This method assumes that `f` uetrieves compile time information and `g` is the fall back
# for the corresponding dynamic method. If the `f(x)` doesn't return `nothing` that means
# the value is known and compile time and returns `static(f(x))`.
@inline function maybe_static(f::F, g::G, x) where {F, G}
    L = f(x)
    if L === nothing
        return g(x)
    else
        return static(L)
    end
end

"""
    eq(x, y)

Equivalent to `!=` but if `x` and `y` are both static returns a `StaticBool.
"""
eq(x::X, y::Y) where {X, Y} = ifelse(is_static(X) & is_static(Y), static, identity)(x == y)
eq(x) = Base.Fix2(eq, x)

"""
    ne(x, y)

Equivalent to `!=` but if `x` and `y` are both static returns a `StaticBool.
"""
ne(x::X, y::Y) where {X, Y} = !eq(x, y)
ne(x) = Base.Fix2(ne, x)

"""
    gt(x, y)

Equivalent to `>` but if `x` and `y` are both static returns a `StaticBool.
"""
gt(x::X, y::Y) where {X, Y} = ifelse(is_static(X) & is_static(Y), static, identity)(x > y)
gt(x) = Base.Fix2(gt, x)

"""
    ge(x, y)

Equivalent to `>=` but if `x` and `y` are both static returns a `StaticBool.
"""
ge(x::X, y::Y) where {X, Y} = ifelse(is_static(X) & is_static(Y), static, identity)(x >= y)
ge(x) = Base.Fix2(ge, x)

"""
    le(x, y)

Equivalent to `<=` but if `x` and `y` are both static returns a `StaticBool.
"""
le(x::X, y::Y) where {X, Y} = ifelse(is_static(X) & is_static(Y), static, identity)(x <= y)
le(x) = Base.Fix2(le, x)

"""
    lt(x, y)

Equivalent to `<` but if `x` and `y` are both static returns a `StaticBool.
"""
lt(x::X, y::Y) where {X, Y} = ifelse(is_static(X) & is_static(Y), static, identity)(x < y)
lt(x) = Base.Fix2(lt, x)

"""
    mul(x) -> Base.Fix2(*, x)
    mul(x, y) ->

Equivalent to `*` but allows for lazy multiplication when passing functions.
"""
mul(x) = Base.Fix2(*, x)

"""
    add(x) -> Base.Fix2(+, x)
    add(x, y) ->

Equivalent to `+` but allows for lazy addition when passing functions.
"""
add(x) = Base.Fix2(+, x)

const Mul{X} = Base.Fix2{typeof(*), X}
const Add{X} = Base.Fix2{typeof(+), X}


_final_isless(c::Int) = c === 1
_final_isless(::StaticInt{N}) where {N} = static(false)
_final_isless(::StaticInt{1}) = static(true)
_isless(c::C, x::Tuple{}, y::Tuple{}) where {C} = _final_isless(c)
function _isless(c::C, x::Tuple, y::Tuple) where {C}
    _isless(icmp(c, x, y), Base.front(x), Base.front(y))
end


function Base.show(io::IO, @nospecialize(x::StaticFloat))
    show(io, MIME"text/plain"(), x)
end
function Base.show(io::IO, ::MIME"text/plain",
                   @nospecialize(x::StaticFloat))
    print(io, "static(" * repr(known(typeof(x))) * ")")
end

end
