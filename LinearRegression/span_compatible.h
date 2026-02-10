#pragma once

#include <cmath>
#include <type_traits>
#include <limits>

#include <span>
#include <concepts>

namespace Helper {
    template <typename T>
        requires std::is_floating_point_v<T>
    constexpr bool nearly_equal(T a, T b, T rel_eps = std::numeric_limits<T>::epsilon(),
        T abs_eps = std::numeric_limits<T>::epsilon())
    {
        T diff = std::fabs(a - b);

        // absoluter Vergleich (nahe 0)
        if (diff <= abs_eps)
            return true;

        // relativer Vergleich
        return diff <= std::max(std::fabs(a), std::fabs(b)) * rel_eps;
    }

    template <class C>
    concept SpanCompatible =
        requires(const C & c)
    {
        { c.data() } -> std::convertible_to<const typename C::value_type*>;
        { c.size() } -> std::convertible_to<std::size_t>;
    }&& std::is_arithmetic_v<typename C::value_type>;

    // Helper: Container -> span
    template <SpanCompatible C>
    auto as_span(const C& c) -> std::span<const typename C::value_type>
    {
        return { c.data(), c.size() };
    }
}
