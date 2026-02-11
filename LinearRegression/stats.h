/**
 * @file stats.h
 * @brief Statistical utility functions for numerical data analysis
 * @author Haasrobertgmxnet
 * @date 2025
 *
 * This header provides fundamental statistical operations used in regression
 * analysis and general numerical computing. All functions are template-based
 * and work with std::span for efficiency.
 *
 * Functions include:
 * - mean: Calculate arithmetic mean
 * - shift: Center data around mean
 * - inner_product: Compute dot product of two vectors
 */

#pragma once

#include <numeric>
#include "span_compatible.h"

namespace Stats {

	/**
	 * @brief Calculates the arithmetic mean (average) of a dataset
	 * @tparam T Numeric type (must be arithmetic: int, float, double, etc.)
	 * @param data Span of data values
	 * @return The mean as T
	 * @throws std::invalid_argument if data is empty
	 *
	 * The arithmetic mean is calculated as: x̄ = (Σxᵢ) / n
	 * where n is the number of elements.
	 *
	 * This is the most common measure of central tendency and represents
	 * the "center of mass" of the distribution.
	 *
	 * Example:
	 * @code
	 * std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
	 * double avg = Stats::mean(data);  // Returns 3.0
	 * @endcode
	 *
	 * @note The result is always returned as double for precision,
	 *       even if input type T is float or int
	 */
	template <class T>
	[[nodiscard]]
	T mean(std::span<const T> data)
	{
		// Compile-time check: ensure T is a numeric type
		static_assert(std::is_arithmetic_v<T>, "mean: T must be arithmetic");

		// Runtime validation: cannot compute mean of empty dataset
		if (data.empty())
			throw std::invalid_argument("mean: data must not be empty");

		// Sum all values and divide by count
		// std::accumulate handles the iteration and summation
		// Initial value 0.0 ensures we get double precision
		return std::accumulate(data.begin(), data.end(), 0.0) / static_cast<T>(data.size());
	}

	/**
	 * @brief Container overload for mean function
	 * @tparam C Container type that is SpanCompatible
	 * @param c Container of numeric values
	 * @return The mean as a double
	 *
	 * This overload allows using any container (vector, array, deque, etc.)
	 * that provides data() and size() methods.
	 *
	 * Example:
	 * @code
	 * std::array<int, 5> arr = {1, 2, 3, 4, 5};
	 * double avg = Stats::mean(arr);  // Works with any compatible container
	 * @endcode
	 */
	template <Helper::SpanCompatible C>
	[[nodiscard]]
	double mean(const C& c)
	{
		return mean(Helper::as_span(c));
	}

	/**
	 * @brief Centers data around its mean (subtracts mean from each value)
	 * @tparam T Numeric type
	 * @param x Span of data values
	 * @return Vector containing centered values: x[i] - mean(x)
	 * @throws std::invalid_argument if data is empty
	 *
	 * This operation is also known as "mean centering" or "demeaning".
	 * It transforms the data so that the new mean is zero.
	 *
	 * Mathematical operation: x'ᵢ = xᵢ - x̄
	 *
	 * Uses:
	 * - Improves numerical stability in regression calculations
	 * - Removes location bias when comparing datasets
	 * - Prerequisite for many statistical calculations (variance, covariance)
	 *
	 * Example:
	 * @code
	 * std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
	 * auto centered = Stats::shift(data);  // Returns {-2, -1, 0, 1, 2}
	 * // New mean is approximately 0
	 * @endcode
	 *
	 * @note Returns a new vector; does not modify the input
	 */
	template <class T>
	[[nodiscard]]
	std::vector<T> shift(std::span<const T> x)
	{
		// Create a copy of the data (we'll modify this copy)
		std::vector<T> data(x.size());
		std::copy(x.begin(), x.end(), data.begin());

		// Validate input
		if (data.empty())
			throw std::invalid_argument("shift: data must not be empty");

		// Calculate the mean of the dataset
		T mean = Stats::mean(data);

		// Subtract mean from each element
		// After this operation, the mean of 'data' will be (approximately) zero
		for (auto& ele : data) {
			ele -= mean;
		}

		return data;
	}

	/**
	 * @brief Container overload for shift function
	 * @tparam C Container type that is SpanCompatible
	 * @param c Container of numeric values
	 * @return Centered values as a vector
	 *
	 * Convenience overload that accepts any compatible container type.
	 */
	template <Helper::SpanCompatible C>
	[[nodiscard]]
	double shift(const C& c)
	{
		return shift(Helper::as_span(c));
	}

	/**
	 * @brief Calculates the inner product (dot product) of two vectors
	 * @tparam T Numeric type for both vectors
	 * @param x First vector
	 * @param y Second vector
	 * @return The inner product as a T: Σ(xᵢ × yᵢ)
	 * @throws std::invalid_argument if vectors have different sizes or size < 2
	 *
	 * The inner product (also called dot product) is defined as:
	 * x · y = Σ(xᵢ × yᵢ) = x₁y₁ + x₂y₂ + ... + xₙyₙ
	 *
	 * This is a fundamental operation in linear algebra and statistics:
	 * - Used to calculate covariance when vectors are centered
	 * - Used to calculate variance when both vectors are the same
	 * - Measures how much two vectors "align" with each other
	 *
	 * Properties:
	 * - Commutative: x · y = y · x
	 * - If vectors are perpendicular (uncorrelated), dot product is 0
	 * - If vectors point in same direction, dot product is positive
	 * - If vectors point in opposite directions, dot product is negative
	 *
	 * Example:
	 * @code
	 * std::vector<double> x = {1.0, 2.0, 3.0};
	 * std::vector<double> y = {4.0, 5.0, 6.0};
	 * double result = Stats::inner_product(x, y);  // Returns 1*4 + 2*5 + 3*6 = 32
	 * @endcode
	 *
	 * @note This implementation uses std::transform_reduce which can be
	 *       parallelized by the compiler for large datasets
	 */
	template <class T>
	[[nodiscard]]
	T inner_product(std::span<const T> x, std::span<const T> y)
	{
		// Compile-time check: ensure both types are arithmetic
		static_assert(std::is_arithmetic_v<T> && std::is_arithmetic_v<T>,
			"covariance: T and U must be arithmetic");

		// Runtime validation: vectors must have same size and at least 2 elements
		if (x.size() != y.size() || x.size() < 2)
			throw std::invalid_argument("covariance: vectors must have same size >= 2");

		// Calculate the dot product using transform_reduce
		// This is the modern C++17 way to compute element-wise operations
		// It can be automatically parallelized by the compiler
		return std::transform_reduce(
			x.begin(), x.end(),  // Range for first vector
			y.begin(),           // Start of second vector (parallel iteration)
			0.0,                 // Initial value for accumulation
			std::plus<>{},       // Reduction operation: sum the products
			[](T a, T b) {       // Transformation: multiply corresponding elements
				return static_cast<T>(a) * static_cast<T>(b);
			}
		);
	}

	/**
	 * @brief Container overload for inner_product function
	 * @tparam C Container type that is SpanCompatible
	 * @param x First container
	 * @param y Second container
	 * @return The inner product as a double
	 *
	 * Convenience overload accepting any compatible containers.
	 *
	 * Example:
	 * @code
	 * std::vector<double> v1 = {1, 2, 3};
	 * std::array<double, 3> v2 = {4, 5, 6};
	 * double result = Stats::inner_product(v1, v2);  // Works with mixed types
	 * @endcode
	 */
	template <Helper::SpanCompatible C>
	[[nodiscard]]
	double inner_product(const C& x, const C& y)
	{
		return inner_product(Helper::as_span(x), Helper::as_span(y));
	}

} // namespace Stats