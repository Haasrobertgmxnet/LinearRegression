#pragma once

//#include <algorithm>
//#include <span>
//#include <numeric>
//#include <cmath>
//#include <stdexcept>
//#include <cstddef>
//#include <type_traits>
//#include <concepts>
//#include <optional>


#include <vector>
#include <random>
#include <iostream>

#include "linreg.h"

int main()
{
    {
        // Boost test
        boost::math::students_t dist(10);
        std::cout << boost::math::quantile(dist, 0.975) << "\n"; // Expected Output: 2.22814
    }

    // test case 1
    {
        // Beispiel 4.5
        auto x = std::vector<double>{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
        auto y = std::vector<double>{ 0.00081, 0.00163, 0.00244, 0.00325, 0.00407, 0.00488 };

        // Ausgabe
        std::cout << "x:\n";
        for (auto v : x) std::cout << v << " ";
        std::cout << "\n\ny:\n";
        for (auto v : y) std::cout << v << " ";
        std::cout << "\n";

		auto res = LinearRegression::fit(x, y);
		auto ci = LinearRegression::ci_slope(res, 0.01);
    }

    // Test case 2
    {
        auto x = std::vector<double>{ 1,2,3,4,5,6,7,8,9,10 };
        auto y = std::vector<double>{ 3.1,5.0,7.2,9.1,10.9,13.05,15.0,16.8,19.2,21.1 };

        auto res = LinearRegression::fit(x, y);
        auto ci = LinearRegression::ci_slope(res, 0.05); // 95%-KI, alpha=0.05

    }
    



}



