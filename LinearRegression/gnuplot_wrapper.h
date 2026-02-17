#pragma once
#include <span>
#include <string_view>

class GnuplotWrapper {
public:
	GnuplotWrapper() {
#ifdef _WIN32
		gp = _popen("gnuplot -persistent", "w");
#else
		gp = popen("gnuplot -persistent", "w");
#endif
	}
	~GnuplotWrapper() {
#ifdef _WIN32
		_pclose(gp);
#else
		pclose(gp);
#endif
	}

	bool is_open() const {
		return gp != nullptr;
	}

	void send_command(const char* cmd) {
		if (gp == nullptr) {
			fprintf(stderr, "Error: Gnuplot not initialized.\n");
			return;
		}
		fprintf(gp, "%s\n", cmd);
		fflush(gp);
	}

	void send_command(std::string_view s) {
		send_command(s.data());
	}

	void plot(const std::span<double>& x, const std::span<double>& y) {
		fprintf(gp,
			"plot "
			"'-' using 1:2:3 with filledcurves title '95%% CI', "
			"'-' with lines lw 2 dashtype 0.8 title 'Regression', "
			"'-' with points pt 7 ps 0.8 title 'Data'\n"
		);
		for (size_t i = 0; i < x.size(); ++i)
			fprintf(gp, "%f %f\n", x[i], y[i]);
		fprintf(gp, "e\n");
	};

private:
	FILE* gp{};
};