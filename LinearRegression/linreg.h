#pragma once

#include <utility>
#include <cmath>
#include <boost/math/distributions/students_t.hpp>
#include "span_compatible.h"
#include "stats.h"


namespace LinearRegression {

	template <class T>
	struct FitResult {
		T beta0;
		T beta1;
		T rho;
		T sxx;
		T syy;
		T sxy;
		T sse;
		std::size_t n;
	};

	
    template <class T>
	[[nodiscard]]
    FitResult<T> fit(std::span<const T> x, std::span<const T> y)
    {
		if (x.size() != y.size() || x.size() < 3) {
			return {}; // oder throw, oder optional
		}
		auto x0 = Stats::shift(x);
		auto y0 = Stats::shift(y);
		auto fitResult = FitResult<T>{};
		fitResult.n = x.size();
		fitResult.sxx = Stats::inner_product(x0, x0);
		if (fitResult.sxx == 0.0) {
			return {}; // keine Steigung definierbar
		}
		fitResult.syy = Stats::inner_product(y0, y0);
		fitResult.sxy = Stats::inner_product(x0, y0);
		fitResult.beta1 = fitResult.sxy / fitResult.sxx;
		fitResult.beta0 = Stats::mean(y) - fitResult.beta1 * Stats::mean(x);
		fitResult.rho = fitResult.sxy / std::sqrt(fitResult.sxx * fitResult.syy);

		fitResult.sse = std::transform_reduce(
			x.begin(), x.end(),
			y.begin(),
			0.0,
			std::plus<>(),
			[&fitResult](const T& xi, const T& yi) {
				const T yi_pred = fitResult.beta0 + fitResult.beta1 * xi;
				const T diff = yi - yi_pred;
				return diff * diff;
			}
		);
		
		return fitResult;
    }

    // Overload: Container-Variante
    template <Helper::SpanCompatible C>
	[[nodiscard]]
    FitResult<double> fit(const C& x, const C& y)
    {
        return fit(Helper::as_span(x), Helper::as_span(y));
    }

	double t_quantile_boost(double p, double m) {
		boost::math::students_t dist(m);
		return boost::math::quantile(dist, p);
	}



	std::pair<double, double>
		ci_slope(const FitResult<double>& fitResult, const double alpha)
	{
		const auto dof = fitResult.n - 2;

		const auto scal = fitResult.syy / fitResult.sxx;
		const auto sb =  std::sqrt(scal * 
			(1.0 - fitResult.beta1 * fitResult.beta1 / scal) / dof
		);

		const auto quantile = t_quantile_boost(1.0 - 0.5 * alpha, dof);
		const auto k = quantile * sb;

		const double lower = fitResult.beta1 - k;
		const double upper = fitResult.beta1 + k;

		return { lower, upper };
	}



} // namespace LinearRegression