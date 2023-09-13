# adopted from https://github.com/SciML/Surrogates.jl/blob/e6aa022e612ac57228506e625c662438d385e69d/src/utils.jl
_expected_dimension(x) = length(x[1])

function _check_dimension(surr, input)
    expected_dim = _expected_dimension(surr.x)
    input_dim = length(input)

    if input_dim != expected_dim
        throw(ArgumentError("This surrogate expects $expected_dim-dimensional inputs, but the input had dimension $input_dim."))
    end
    return nothing
end

# delete by a key from a NamedTuple
# adopted from https://discourse.julialang.org/t/remove-a-field-from-a-namedtuple/34664
function delete(nt::NamedTuple{names}, key) where {names}
    NamedTuple{filter(x -> x !== key, names)}(nt)
end
