# delete by a key from a NamedTuple
# adopted from https://discourse.julialang.org/t/remove-a-field-from-a-namedtuple/34664
function delete(nt::NamedTuple{names}, key) where {names}
    NamedTuple{filter(x -> x !== key, names)}(nt)
end
