#pragma once

#include <algorithm>   // std::max
#include <cmath>       // std::fabs
#include <concepts>    // std::floating_point, std::convertible_to
#include <cstddef>     // std::size_t
#include <limits>      // std::numeric_limits
#include <span>        // std::span

namespace Helper {

    /**
     * @brief Compares two floating-point values for approximate equality.
     *
     * Uses a combination of absolute and relative comparison to account for
     * floating-point rounding errors.
     *
     * @param a First value.
     * @param b Second value.
     * @param rel_eps Relative tolerance (default: machine epsilon).
     * @param abs_eps Absolute tolerance (default: machine epsilon).
     * @return true if values are approximately equal, false otherwise.
     */
    template <std::floating_point T>
    [[nodiscard]]
    constexpr bool nearly_equal(
        T a, T b,
        T rel_eps = std::numeric_limits<T>::epsilon(),
        T abs_eps = std::numeric_limits<T>::epsilon())
    {
        const T diff = std::fabs(a - b);

        // Absolute comparison (near zero)
        if (diff <= abs_eps)
            return true;

        // Relative comparison
        return diff <= std::max(std::fabs(a), std::fabs(b)) * rel_eps;
    }

    /**
     * @brief Concept for containers that can be viewed as a read-only std::span.
     *
     * Requirements:
     * - Provides data() returning a pointer convertible to const value_type*
     * - Provides size() convertible to std::size_t
     * - value_type must be a floating-point type
     */
    template <class C>
    concept SpanCompatible =
        requires (const C & c) {
            { c.data() } -> std::convertible_to<const typename C::value_type*>;
            { c.size() } -> std::convertible_to<std::size_t>;
    }
    && std::floating_point<typename C::value_type>;

    /**
     * @brief Creates a read-only std::span view from a SpanCompatible container.
     * @param c Input container.
     * @return std::span over the container's elements.
     */
    template <SpanCompatible C>
    [[nodiscard]]
    constexpr auto as_span(const C& c) -> std::span<const typename C::value_type>
    {
        return { c.data(), c.size() };
    }

} // namespace Helper
