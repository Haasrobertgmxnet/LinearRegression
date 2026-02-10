#pragma once
#include <numeric>
#include "span_compatible.h"

namespace Stats {
    template <class T>
    [[nodiscard]]
    double mean(std::span<const T> data)
    {
        static_assert(std::is_arithmetic_v<T>, "mean: T must be arithmetic");
        if (data.empty())
            throw std::invalid_argument("mean: data must not be empty");

        return std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());
    }

    // Overload: nimmt vector/array/etc direkt
    template <Helper::SpanCompatible C>
    [[nodiscard]]
    double mean(const C& c)
    {
        return mean(Helper::as_span(c));
    }

    template <class T>
    [[nodiscard]]
    std::vector<T>  shift(std::span<const T> x)
    {
        std::vector<T> data(x.size());
        std::copy(x.begin(), x.end(), data.begin());
        if (data.empty())
            throw std::invalid_argument("shift: data must not be empty");
		T mean = Stats::mean(data);
        for (auto& ele : data) {
            ele -= mean;

        }
        return data;
	}

    // Overload: nimmt vector/array/etc direkt
    template <Helper::SpanCompatible C>
    [[nodiscard]]
    double shift(const C& c)
    {
        return shift(Helper::as_span(c));
    }

    template <class T>
    [[nodiscard]]
    double inner_product(std::span<const T> x, std::span<const T> y)
    {
        static_assert(std::is_arithmetic_v<T> && std::is_arithmetic_v<T>,
            "covariance: T and U must be arithmetic");

        if (x.size() != y.size() || x.size() < 2)
            throw std::invalid_argument("covariance: vectors must have same size >= 2");

        return std::transform_reduce(
            x.begin(), x.end(), y.begin(),
            0.0,
            std::plus<>{},
            [](T a, T b) { return static_cast<double>(a) * static_cast<double>(b); }
        );
    }

    // Overload: Container-Variante
    template <Helper::SpanCompatible C>
    [[nodiscard]]
    double inner_product(const C& x, const C& y)
    {
        return inner_product(Helper::as_span(x), Helper::as_span(y));
    }

}