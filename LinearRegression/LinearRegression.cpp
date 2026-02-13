#pragma once

#include <vector>
#include <cstdio>
#include <cmath>

// Öffnet gnuplot per Pipe (Windows + Linux)
FILE* open_gnuplot()
{
#ifdef _WIN32
    return _popen("gnuplot -persistent", "w");
#else
    return popen("gnuplot -persistent", "w");
#endif
}

void close_gnuplot(FILE* gp)
{
#ifdef _WIN32
    _pclose(gp);
#else
    pclose(gp);
#endif
}


#include <span>
#include <random>
#include <iostream>

#include "linreg.h"

int plot_chart(const std::span<double>& x, const std::span<double>& y)
{
    auto res = LinearRegression::fit(x, y);
    auto ci_pair = LinearRegression::ci_slope(res, 0.01); // 95%-KI, alpha=0.05

    // Einfaches konstantes Konfidenzintervall (Demo)
	double ci = 0.5 * (ci_pair.second - ci_pair.first); // Halbbreite des Intervalls

    // ci = 0.3;

    // Lineare Regression (Demo-Werte)
    double a = res.beta0;   // Intercept
    double b = res.beta1;   // Steigung

    FILE* gp = open_gnuplot();
    if (!gp) return 1;

    // Terminal: qt ist interaktiv; unterstützt meist Transparenz
    fprintf(gp, "set term qt\n");  // optional, je nach System
    fprintf(gp, "set grid\n");
    fprintf(gp, "set key left top\n");

    // Transparenz-Style für Füllungen (alpha 0..1)
    fprintf(gp, "set style fill transparent solid 0.20 noborder\n");
    fprintf(gp, "set border linewidth 1\n");

    // Plot: erst CI-Band, dann Regression, dann Punkte (oben)
    fprintf(gp,
        "plot "
        "'-' using 1:2:3 with filledcurves title '95%% CI', "
        "'-' with lines lw 2 dashtype 0.8 title 'Regression', "
        "'-' with points pt 7 ps 0.8 title 'Data'\n"
    );

    // 1) Konfidenzintervall (x, lower, upper)
    for (double xi : x) {
        double yhat = a + b * xi;
        fprintf(gp, "%f %f %f\n", xi, yhat - ci, yhat + ci);
    }
    fprintf(gp, "e\n");

    // 2) Regressionsgerade
    for (double xi : x)
        fprintf(gp, "%f %f\n", xi, a + b * xi);
    fprintf(gp, "e\n");

    // 3) Punktewolke
    for (size_t i = 0; i < x.size(); ++i)
        fprintf(gp, "%f %f\n", x[i], y[i]);
    fprintf(gp, "e\n");


    fflush(gp);
    close_gnuplot(gp);
}

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
        // auto y = std::vector<double>{ 3.1,5.0,7.2,9.1,10.9,13.05,15.0,16.8,19.2,21.1 };
        auto y = std::vector<double>{ 3.1,5.0,7.2,9.1,10.0,13.2,15.5,16.5,19.0,21.3 };

        auto res = LinearRegression::fit(x, y);
        auto ci = LinearRegression::ci_slope(res, 0.05); // 95%-KI, alpha=0.05

        plot_chart(x, y);

    }
    
    return 0;


}



