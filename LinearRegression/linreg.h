/**
 * @file LinearRegression.h
 * @brief Modern C++20 implementation of simple linear regression
 * @author Haasrobertgmxnet
 * @date 2025
 *
 * This header provides a template-based linear regression implementation
 * using the least squares method. It leverages modern C++20 features
 * including std::span, concepts, and structured bindings.
 *
 * Key features:
 * - Template-based design for flexibility with numeric types
 * - std::span support for efficient memory usage
 * - Boost.Math integration for statistical distributions
 * - Comprehensive statistical output including confidence intervals
 */

#pragma once

#include <utility>
#include <cmath>
#include <boost/math/distributions/students_t.hpp>
#include "span_compatible.h"
#include "stats.h"

namespace LinearRegression {

	/**
	 * @brief Result structure containing all regression statistics
	 * @tparam T Numeric type for regression coefficients (typically double)
	 *
	 * This structure holds all computed values from a linear regression fit.
	 * It contains both the model parameters (beta0, beta1) and diagnostic
	 * statistics (rho, sse, sums of squares).
	 */
	template <class T>
	struct FitResult {
		T beta0;        ///< y-intercept (β₀): The predicted y value when x = 0
		T beta1;        ///< Slope coefficient (β₁): The change in y per unit change in x
		T rho;          ///< Pearson correlation coefficient (ρ): Measures linear relationship strength (-1 to 1)
		T sxx;          ///< Sum of squared deviations of x: Σ(xᵢ - x̄)²
		T syy;          ///< Sum of squared deviations of y: Σ(yᵢ - ȳ)²
		T sxy;          ///< Sum of products of deviations: Σ(xᵢ - x̄)(yᵢ - ȳ)
		T sse;          ///< Sum of squared errors (residuals): Σ(yᵢ - ŷᵢ)²
		std::size_t n;  ///< Number of data points used in the fit
	};

	/**
	 * @brief Fits a linear regression model to data using least squares
	 * @tparam T Numeric type (must be arithmetic, typically float or double)
	 * @param x Independent variable values (features)
	 * @param y Dependent variable values (targets)
	 * @return FitResult<T> containing all regression statistics
	 *
	 * This function implements the least squares method to find the best-fitting
	 * linear relationship y = β₀ + β₁x between the variables.
	 *
	 * Algorithm:
	 * 1. Center both variables around their means (for numerical stability)
	 * 2. Calculate sums of squares (Sxx, Syy) and cross-products (Sxy)
	 * 3. Compute slope: β₁ = Sxy / Sxx
	 * 4. Compute intercept: β₀ = ȳ - β₁x̄
	 * 5. Calculate correlation: ρ = Sxy / √(Sxx × Syy)
	 * 6. Compute sum of squared errors (SSE) for model quality assessment
	 *
	 * @note Returns empty FitResult if:
	 *       - x and y have different sizes
	 *       - Fewer than 3 data points provided
	 *       - All x values are identical (Sxx = 0)
	 *
	 * Example:
	 * @code
	 * std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
	 * std::vector<double> y = {2.1, 3.9, 6.2, 7.8, 10.1};
	 * auto result = LinearRegression::fit(x, y);
	 * std::cout << "y = " << result.beta0 << " + " << result.beta1 << "x\n";
	 * @endcode
	 */
	template <typename T>
		requires std::is_floating_point_v<T>
	[[nodiscard]]  // Prevents accidentally discarding the result
	FitResult<T> fit(std::span<const T> x, std::span<const T> y)
	{
		// Validate input: ensure same size and minimum data points
		if (x.size() != y.size() || x.size() < 3) {
			return {}; // Return default-constructed (empty) result
		}

		// Center both variables around their means for numerical stability
		// This prevents potential overflow/underflow with very large values
		auto x0 = Stats::shift(x);  // x0[i] = x[i] - mean(x)
		auto y0 = Stats::shift(y);  // y0[i] = y[i] - mean(y)

		// Initialize result structure
		auto fitResult = FitResult<T>{};
		fitResult.n = x.size();

		// Calculate sum of squared deviations for x
		// Sxx = Σ(xᵢ - x̄)² measures spread/variance of x
		fitResult.sxx = Stats::inner_product(x0, x0);

		// Check for degenerate case: all x values are identical
		// If Sxx = 0, slope is undefined (division by zero)
		if (fitResult.sxx == 0.0) {
			return {}; // Cannot fit a line when x doesn't vary
		}

		// Calculate sum of squared deviations for y
		// Syy = Σ(yᵢ - ȳ)² measures spread/variance of y
		fitResult.syy = Stats::inner_product(y0, y0);

		// Calculate sum of products of deviations
		// Sxy = Σ(xᵢ - x̄)(yᵢ - ȳ) measures covariance between x and y
		fitResult.sxy = Stats::inner_product(x0, y0);

		// Calculate slope coefficient using least squares formula
		// β₁ = Sxy / Sxx
		// This gives the change in y per unit change in x
		fitResult.beta1 = fitResult.sxy / fitResult.sxx;

		// Calculate intercept using the formula: β₀ = ȳ - β₁x̄
		// The regression line always passes through the point (x̄, ȳ)
		fitResult.beta0 = Stats::mean(y) - fitResult.beta1 * Stats::mean(x);

		// Calculate Pearson correlation coefficient
		// ρ = Sxy / √(Sxx × Syy)
		// Range: -1 (perfect negative) to +1 (perfect positive)
		// 0 indicates no linear relationship
		fitResult.rho = fitResult.sxy / std::sqrt(fitResult.sxx * fitResult.syy);

		// Calculate sum of squared errors (SSE)
		// SSE = Σ(yᵢ - ŷᵢ)² where ŷᵢ = β₀ + β₁xᵢ
		// This measures how well the model fits the data
		// Lower SSE indicates better fit
		fitResult.sse = std::transform_reduce(
			x.begin(), x.end(),   // Range for x values
			y.begin(),            // Range for y values (parallel iteration)
			0.0,                  // Initial value for accumulation
			std::plus<>(),        // Reduction operation (sum)
			[&fitResult](const T& xi, const T& yi) {
				// For each data point, calculate predicted value
				const T yi_pred = fitResult.beta0 + fitResult.beta1 * xi;
				// Calculate residual (error)
				const T diff = yi - yi_pred;
				// Return squared error
				return diff * diff;
			}
		);

		return fitResult;
	}

	/**
	 * @brief Container overload for fit function
	 * @tparam C Container type that is SpanCompatible (vector, array, etc.)
	 * @param x Container of independent variable values
	 * @param y Container of dependent variable values
	 * @return FitResult<double> containing all regression statistics
	 *
	 * This overload allows passing any container that can be converted to
	 * std::span, making the API more convenient for common use cases.
	 *
	 * Example:
	 * @code
	 * std::vector<double> x = {1, 2, 3, 4};
	 * std::array<double, 4> y = {2, 4, 6, 8};
	 * auto result = LinearRegression::fit(x, y);  // Works with different container types
	 * @endcode
	 */
	template <Helper::SpanCompatible C>
	[[nodiscard]]
	FitResult<double> fit(const C& x, const C& y)
	{
		return fit(Helper::as_span(x), Helper::as_span(y));
	}

	/**
	 * @brief Calculates the t-distribution quantile using Boost.Math
	 * @param p Probability level (e.g., 0.975 for upper 2.5% tail)
	 * @param m Degrees of freedom
	 * @return The t-value at the specified probability level
	 *
	 * This function wraps Boost's Student's t-distribution quantile calculation.
	 * It's used internally for computing confidence intervals.
	 *
	 * The Student's t-distribution is used (instead of normal distribution)
	 * because it accounts for estimation uncertainty when sample size is small.
	 * As degrees of freedom increase, it approaches the normal distribution.
	 */
	template <typename T>
		requires std::is_floating_point_v<T>
	T t_quantile(T p, T m) {
		boost::math::students_t_distribution<T> dist(m);
		return boost::math::quantile(dist, p);
	}

	/**
	 * @brief Calculates confidence interval for the slope coefficient
	 * @param fitResult The result from a previous fit() call
	 * @param alpha Significance level (e.g., 0.05 for 95% confidence)
	 * @return Pair of (lower bound, upper bound) for the slope
	 *
	 * Template version that preserves the precision of the input type.
	 * This is the preferred implementation as it:
	 * - Matches the type used in fit()
	 * - Preserves numerical precision
	 * - Provides consistent API design



	 * Computes the confidence interval for β₁ using the formula:
	 * CI = β₁ ± t(1-α/2, n-2) × SE(β₁)
	 *
	 * Where:
	 * - t(1-α/2, n-2) is the t-distribution quantile
	 * - SE(β₁) is the standard error of the slope
	 * - n-2 degrees of freedom (we estimated 2 parameters: β₀ and β₁)
	 *
	 * The standard error is calculated as:
	 * SE(β₁) = √[Syy/Sxx × (1 - β₁²/(Syy/Sxx)) / (n-2)]
	 *
	 * Example:
	 * @code
	 * auto result = LinearRegression::fit(x, y);
	 * auto [lower, upper] = LinearRegression::ci_slope(result, 0.05);
	 * std::cout << "95% CI: [" << lower << ", " << upper << "]\n";
	 * @endcode
	 *
	 * @note Common alpha values:
	 *       - 0.01 for 99% confidence
	 *       - 0.05 for 95% confidence (most common)
	 *       - 0.10 for 90% confidence
	 */
	template <typename T>
		requires std::is_floating_point_v<T>
	[[nodiscard]]
	std::pair<T, T> ci_slope(const FitResult<T>& fitResult, const T alpha)
	{
		// Degrees of freedom: n - 2
		// We lose 2 degrees because we estimated β₀ and β₁
		const auto dof = static_cast<T>(fitResult.n - 2);

		// Scaling factor for standard error calculation
		const auto scal = fitResult.syy / fitResult.sxx;

		// Calculate standard error of the slope (SE(β₁))
		// This measures the uncertainty in our slope estimate
		// Formula: SE(β₁) = √[scal × (1 - β₁²/scal) / dof]
		const auto sb = std::sqrt(scal *
			(T{ 1 } - fitResult.beta1 * fitResult.beta1 / scal) / dof
		);

		// Get the critical t-value for the desired confidence level
		// For 95% CI with α=0.05, we need t at probability 0.975
		// (two-tailed test, so we use 1 - α/2)
		// Get the critical t-value using templated quantile function
		const auto quantile = t_quantile(T{ 1 } - T{ 0.5 } * alpha, dof);

		// Calculate margin of error: ME = t × SE(β₁)
		const auto k = quantile * sb;

		// Construct confidence interval: [β₁ - ME, β₁ + ME]
		const T lower = fitResult.beta1 - k;
		const T upper = fitResult.beta1 + k;

		return { lower, upper };
	}

	/**
	 * @brief Double-precision specialization (for backward compatibility)
	 * @param fitResult Result from fit() with double precision
	 * @param alpha Significance level
	 * @return Confidence interval bounds
	 *
	 * This specialization is kept for backward compatibility and convenience.
	 * It's equivalent to calling the template version with T=double.
	 */
	[[nodiscard]]
	inline std::pair<double, double>
		ci_slope(const FitResult<double>& fitResult, const double alpha)
	{
		return ci_slope<double>(fitResult, alpha);
	}

	/**
 * @brief Calculates the coefficient of determination (R²)
 * @tparam T Floating-point type (float, double, long double)
 * @param fitResult The result from a previous fit() call
 * @return R² value ranging from 0 to 1
 *
 * The coefficient of determination (R²) measures the proportion of variance
 * in the dependent variable that is predictable from the independent variable.
 * It is computed as the square of the Pearson correlation coefficient.
 *
 * Mathematical formula: R² = ρ²
 *
 * Interpretation:
 * - R² = 1.0: Perfect fit - all data points lie exactly on the regression line
 * - R² = 0.9-0.99: Excellent fit - strong linear relationship
 * - R² = 0.7-0.89: Good fit - substantial linear relationship
 * - R² = 0.4-0.69: Moderate fit - noticeable linear relationship
 * - R² = 0.0-0.39: Weak fit - little to no linear relationship
 * - R² = 0.0: No linear relationship
 *
 * Note: R² can also be calculated as 1 - (SSE/SST) where:
 * - SSE = sum of squared errors (residuals)
 * - SST = total sum of squares
 *
 * This implementation uses the simpler formula R² = ρ² which is
 * mathematically equivalent for simple linear regression.
 *
 * Example:
 * @code
 * auto result = LinearRegression::fit(x, y);
 * double r_squared = LinearRegression::coeff_of_determination(result);
 * std::cout << "R² = " << r_squared << std::endl;
 *
 * if (r_squared > 0.95) {
 *     std::cout << "Excellent fit!" << std::endl;
 * }
 * @endcode
 *
 * @note This function is marked [[nodiscard]] to prevent accidentally
 *       ignoring the calculated R² value
 */
	template <typename T>
		requires std::is_floating_point_v<T>
	[[nodiscard]]
	T coeff_of_determination(const FitResult<T>& fitResult)
	{
		// R² = ρ² (square of correlation coefficient)
		// This gives the proportion of variance explained by the model
		return fitResult.rho * fitResult.rho;
	}

	/**
	 * @brief Double-precision specialization for coefficient of determination
	 * @param fitResult Result from fit() with double precision
	 * @return R² value as double
	 *
	 * This inline specialization provides backward compatibility and convenience
	 * for the common case of double-precision regression. It simply forwards
	 * to the template version with T=double.
	 *
	 * Example:
	 * @code
	 * FitResult<double> result = LinearRegression::fit(x, y);
	 * double r_squared = LinearRegression::coeff_of_determination(result);
	 * @endcode
	 */
	[[nodiscard]]
	inline double coeff_of_determination(const FitResult<double>& fitResult)
	{
		return coeff_of_determination<double>(fitResult);
	}

} // namespace LinearRegression
