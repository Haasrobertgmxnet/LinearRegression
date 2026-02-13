/**
 * @file stats.h
 * @brief Statistical utility functions (mean-centering, dot product) on spans/containers.
 *
 * Notes:
 * - Functions are constrained to floating-point types.
 * - Parallel execution uses std::execution::par and may yield tiny FP rounding differences.
 */
#pragma once
#include <algorithm>
#include <concepts>
#include <execution>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "span_compatible.h"

namespace Stats {

	/**
	 * @brief Arithmetic mean of a non-empty dataset.
	 * @param data Input values.
	 * @return Mean value.
	 * @throws std::invalid_argument if data is empty.
	 */
	template <std::floating_point T>
	[[nodiscard]]
	T mean(std::span<const T> data)
	{
		// Runtime validation: cannot compute mean of empty dataset
		if (data.empty())
			throw std::invalid_argument("mean: data must not be empty");

		// using Acc = std::common_type_t<T, double>;

		// Sum all values and divide by count
		// std::reduce handles the iteration and summation
		// Initial value T{0} ensures we get floating point precision
		auto sum = std::reduce(std::execution::par,
			data.begin(), data.end(),
			T{ 0 }, std::plus<>{});
		return sum / static_cast<T>(data.size());
	}

	/// @brief Convenience overload: accepts any SpanCompatible container.
	template <Helper::SpanCompatible C>
	[[nodiscard]]
	auto mean(const C& c)
	{
		return mean(Helper::as_span(c));
	}

	/**
	 * @brief Mean-centers a dataset (x[i] -= mean(x)).
	 * @param x Input values.
	 * @return New vector containing centered values.
	 * @throws std::invalid_argument if x is empty.
	 */
	template <std::floating_point T>
	[[nodiscard]]
	std::vector<T> shift(std::span<const T> x)
	{
		// Create a copy of the data (we'll modify this copy)
		std::vector<T> data(x.begin(), x.end());

		// Validate input
		if (data.empty())
			throw std::invalid_argument("shift: data must not be empty");

		// Calculate the mean of the dataset
		auto m = Stats::mean(data);

		// Subtract mean from each element
		// After this operation, the mean of 'data' will be (approximately) zero
		std::transform(std::execution::par,
			data.begin(), data.end(), data.begin(),
			[m](T v) { return v - m; });

		return data;
	}

	/// @brief Convenience overload: accepts any SpanCompatible container.
	template <Helper::SpanCompatible C>
	[[nodiscard]]
	auto shift(const C& c)
	{
		return shift(Helper::as_span(c));
	}

	/**
	 * @brief Dot product of two equally-sized vectors.
	 * @param x First vector.
	 * @param y Second vector.
	 * @return Sum of x[i] * y[i].
	 * @throws std::invalid_argument if sizes differ or size < 2.
	 *
	 * Uses std::transform_reduce with std::execution::par.
	 */
	template <std::floating_point T>
	[[nodiscard]]
	T inner_product(std::span<const T> x, std::span<const T> y)
	{

		// Runtime validation: vectors must have same size and at least 2 elements
		if (x.size() != y.size() || x.size() < 2)
			throw std::invalid_argument("inner_product: vectors must have same size >= 2");

		// Calculate the dot product using transform_reduce
		// This is the modern C++17 way to compute element-wise operations
		return std::transform_reduce(std::execution::par,
			x.begin(), x.end(),
			y.begin(),
			T{},
			std::plus<>{},
			std::multiplies<>{});
	}

	/// @brief Convenience overload: accepts any SpanCompatible container.
	template <Helper::SpanCompatible C>
	[[nodiscard]]
	auto inner_product(const C& x, const C& y)
	{
		return inner_product(Helper::as_span(x), Helper::as_span(y));
	}

} // namespace Stats